// A particle push that files particles per cell inside tiles and registers every particle
// that changed cell (an atomic counter + slot per old cell), launched as a mutable __device__
// lambda capturing the pusher by value. The same pusher runs on the host over a copy of the
// same particles: the moves registered on the device must equal the host's, and every moved
// particle must be registered.
//
// Passes on NVIDIA GPUs. Under PantheonSim's CUDA backend (simulated nvidia/h100, rtx5090) about
// a third of the registrations made in the out of line call (per_particle_still_in_ghost_box) are
// missing; with that function inlined it passes. Any backend: cuda, rocm and cpu profiles; see
// .github/workflows/gpu-sim.yml.

#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HD __host__ __device__
#define NOINLINE __attribute__((noinline))  // __noinline__ in CUDA / HIP, also for g++

using u64 = unsigned long long;

template <typename T>
using ManagedVector = std::vector<T, mkn::gpu::ManagedAllocator<T>>;

static constexpr int dim = 2, cells = 20, tile_cells = 10, ghost = 2, ppc = 6, cap = 128;
static constexpr int n_tiles_1d = cells / tile_cells, n_tiles = n_tiles_1d * n_tiles_1d;
static constexpr int tile_g = tile_cells + 2 * ghost;  // a tile's ghost box width
static constexpr int set_g = cells + 2 * ghost;        // the tile set's ghost box width
static constexpr int tile_cells_g = tile_g * tile_g;   // a tile's cells, ghosts included
static constexpr int set_cells_g = set_g * set_g;

template <typename T>
struct Arr {  // std::array without its host only operator[] (nvcc)
  T v[dim];
  HD T& operator[](int const i) { return v[i]; }
  HD T const& operator[](int const i) const { return v[i]; }
};

using Cell = Arr<int>;

HD inline bool equals(Cell const& a, Cell const& b) {
  for (int d = 0; d < dim; ++d)
    if (a[d] != b[d]) return false;
  return true;
}

struct Box {
  Cell lower, upper;
  HD bool contains(Cell const& c) const {
    for (int d = 0; d < dim; ++d)
      if (c[d] < lower[d] or c[d] > upper[d]) return false;
    return true;
  }
};

struct Particle {
  double weight;
  Cell iCell_;
  Arr<double> delta_;
  double v_[3];
  HD auto& iCell() { return iCell_; }
  HD auto& iCell() const { return iCell_; }
  HD auto& delta() { return delta_; }
  HD auto& v() { return v_; }
};

struct EM {  // uniform fields, what an interpolation would give
  double E[3], B[3];
};

struct Layout {
  Arr<double> halfDtOverDl;
};

struct Tracker {  // the cell and tile a particle comes from, copied before the push
  Cell icell;
  Arr<std::uint32_t> tile_cell;
  HD int tile() const { return tile_cell[0] * n_tiles_1d + tile_cell[1]; }
};

HD inline u64 increment_return_old(u64& t) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || MKN_GPU_CPU
  return atomicAdd(&t, u64{1});  // the device's, or the cpu backend's
#else
  return __atomic_fetch_add(&t, u64{1}, __ATOMIC_SEQ_CST);
#endif
}

// a tile: per cell, its particles and its departures, in managed memory, reached by pointer
struct Tile {
  Box box, ghost_box;
  Particle* particles;  // tile_cells_g * cap: a cell's particles at [cell * cap, + size)
  u64* sizes;           // tile_cells_g
  u64* gap_idx;         // tile_cells_g: departures counted per old cell
  u64* gaps;            // tile_cells_g * cap: their indices

  HD int local(Cell const& c) const {
    return (c[0] - ghost_box.lower[0]) * tile_g + (c[1] - ghost_box.lower[1]);
  }

  HD Tile& move_check(Tracker const& pt, u64 const idx, Particle& p) {
    auto const& newcell = p.iCell();
    if (equals(newcell, pt.icell)) return *this;  // no change
    auto const old = local(pt.icell);
    gaps[old * cap + increment_return_old(gap_idx[old])] = idx;
    return *this;
  }
};

// the tiles and the departures to another tile
struct TileSet {
  Box box, ghost_box;
  Tile* tiles;    // n_tiles
  u64* gap_idx;   // set_cells_g
  u64* gaps;      // set_cells_g * cap
  u64* add_into;  // set_cells_g: arrivals per new cell

  HD int local(Cell const& c) const {
    return (c[0] - ghost_box.lower[0]) * set_g + (c[1] - ghost_box.lower[1]);
  }

  HD TileSet& move_check(Tracker const& pt, u64 const idx, Particle& p) {
    auto const& newcell = p.iCell();
    if (equals(newcell, pt.icell)) return *this;

    auto const leave = [&]() {
      auto const old = local(pt.icell);
      gaps[old * cap + increment_return_old(gap_idx[old])] = idx;
    };

    if (box.contains(newcell) and not tiles[pt.tile()].box.contains(newcell)) {
      leave();
      increment_return_old(add_into[local(newcell)]);
      return *this;
    }
    tiles[pt.tile()].move_check(pt, idx, p);  // within the tile, or out of the domain
    return *this;
  }
};

struct Pusher {
  TileSet set;
  Layout layout;
  EM em;
  double coef1;

  HD static Cell advance(Particle& p, Arr<double> const& halfDtOverDl) {
    auto newCell = p.iCell();
    for (int iDim = 0; iDim < dim; ++iDim) {
      double delta = p.delta()[iDim] + halfDtOverDl[iDim] * p.v()[iDim];
      double iCell = std::floor(delta);
      p.delta()[iDim] = delta - iCell;
      newCell[iDim] = static_cast<int>(iCell + p.iCell()[iDim]);
    }
    return newCell;
  }

  HD static void accelerate(Particle& p, EM const& eb, double const coef1) {  // Boris
    double const one = 1, two = 2;
    double const velx1 = p.v()[0] + coef1 * eb.E[0];
    double const vely1 = p.v()[1] + coef1 * eb.E[1];
    double const velz1 = p.v()[2] + coef1 * eb.E[2];
    double const rx = coef1 * eb.B[0], ry = coef1 * eb.B[1], rz = coef1 * eb.B[2];
    double const rx2 = rx * rx, ry2 = ry * ry, rz2 = rz * rz;
    double const rxry = rx * ry, rxrz = rx * rz, ryrz = ry * rz;
    double const invDet = one / (one + rx2 + ry2 + rz2);
    double const mxx = one + rx2 - ry2 - rz2, mxy = two * (rxry + rz), mxz = two * (rxrz - ry);
    double const myx = two * (rxry - rz), myy = one + ry2 - rx2 - rz2, myz = two * (ryrz + rx);
    double const mzx = two * (rxrz + ry), mzy = two * (ryrz - rx), mzz = one + rz2 - rx2 - ry2;
    p.v()[0] = (mxx * velx1 + mxy * vely1 + mxz * velz1) * invDet + coef1 * eb.E[0];
    p.v()[1] = (myx * velx1 + myy * vely1 + myz * velz1) * invDet + coef1 * eb.E[1];
    p.v()[2] = (mzx * velx1 + mzy * vely1 + mzz * velz1) * invDet + coef1 * eb.E[2];
  }

  // out of line (a PTX .func): every argument by reference, the tracker a temporary of the
  //  caller. Inlined, the simulator gets it right.
  HD NOINLINE void per_particle_still_in_ghost_box(Particle& particle, Layout const& layout,
                                                   Tracker& pt, u64& pidx, EM const& em) {
    accelerate(particle, em, coef1);
    particle.iCell() = advance(particle, layout.halfDtOverDl);
    set.move_check(pt, pidx, particle);
  }

  HD void per_particle(Particle& particle, Layout const& layout, Tracker&& pt, u64& pidx,
                       EM const& em) {
    particle.iCell() = advance(particle, layout.halfDtOverDl);
    per_particle_still_in_ghost_box(particle, layout, pt, pidx, em);
  }

  // a cell's particles, strided by ws threads from tidx
  HD void per_cell(Tile& tile, int const cell, unsigned const tidx, unsigned const ws) {
    auto const t = static_cast<std::uint32_t>(&tile - set.tiles);
    for (u64 pid = tidx; pid < tile.sizes[cell]; pid += ws) {
      auto& p = tile.particles[cell * cap + pid];
      per_particle(p, layout, Tracker{p.iCell(), {t / n_tiles_1d, t % n_tiles_1d}}, pid, em);
    }
  }

  void on_host() {
    for (int t = 0; t < n_tiles; ++t)
      for (int c = 0; c < tile_cells_g; ++c)
        for (unsigned tidx = 0; tidx < 32; ++tidx) per_cell(set.tiles[t], c, tidx, 32);
  }

  void on_device() {        // block = (cell, tile), the block's threads stride the cell
    unsigned threads = 32;  // a runtime value (a warp), captured
    mkn::gpu::Stream stream;
    mkn::gpu::DLauncher{dim3(tile_cells_g, n_tiles), dim3(threads)}
        .stream(stream,
                [self = *this, threads] __device__() mutable {
                  auto& tile = self.set.tiles[blockIdx.y];
                  self.per_cell(tile, blockIdx.x, threadIdx.x, threads);
                })
        .sync();
  }
};

struct Storage {  // owns what the views point to
  ManagedVector<Tile> tiles;
  ManagedVector<Particle> particles;
  ManagedVector<u64> counters;
  Pusher pusher{};

  explicit Storage(unsigned const seed)
      : tiles(n_tiles),
        particles(n_tiles * tile_cells_g * cap),
        counters(n_tiles * tile_cells_g * (2 + cap) + set_cells_g * (2 + cap), 0) {
    std::srand(seed);
    auto* c = counters.data();
    auto take = [&](std::size_t const n) { return std::exchange(c, c + n); };

    auto& set = pusher.set;
    set.box = Box{{0, 0}, {cells - 1, cells - 1}};
    set.ghost_box = Box{{-ghost, -ghost}, {cells - 1 + ghost, cells - 1 + ghost}};
    set.tiles = tiles.data();
    set.gap_idx = take(set_cells_g);
    set.add_into = take(set_cells_g);
    set.gaps = take(set_cells_g * cap);
    pusher.layout.halfDtOverDl = {{0.25, 0.25}};
    pusher.em = EM{{0.1, -0.2, 0.05}, {0.3, 0.2, 1.0}};
    pusher.coef1 = 0.25;

    auto rnd = [] { return double(std::rand()) / RAND_MAX; };
    for (int tx = 0; tx < n_tiles_1d; ++tx)
      for (int ty = 0; ty < n_tiles_1d; ++ty) {
        auto const ti = tx * n_tiles_1d + ty;
        auto& t = tiles[ti];
        t.box = Box{{tx * tile_cells, ty * tile_cells},
                    {tx * tile_cells + tile_cells - 1, ty * tile_cells + tile_cells - 1}};
        t.ghost_box = Box{{t.box.lower[0] - ghost, t.box.lower[1] - ghost},
                          {t.box.upper[0] + ghost, t.box.upper[1] + ghost}};
        t.particles = particles.data() + ti * tile_cells_g * cap;
        t.sizes = take(tile_cells_g);
        t.gap_idx = take(tile_cells_g);
        t.gaps = take(tile_cells_g * cap);
        for (int x = t.box.lower[0]; x <= t.box.upper[0]; ++x)
          for (int y = t.box.lower[1]; y <= t.box.upper[1]; ++y) {
            int const lc = t.local({{x, y}});
            for (int i = 0; i < ppc; ++i)
              t.particles[lc * cap + i] =
                  Particle{1., {{x, y}}, {{rnd(), rnd()}}, {4 * rnd() - 2, 4 * rnd() - 2, rnd()}};
            t.sizes[lc] = ppc;
          }
      }
  }

  Storage(Storage const&) = delete;
  Storage& operator=(Storage const&) = delete;
};

struct Counts {
  u64 moved = 0, tile_gaps = 0, set_gaps = 0, add_into = 0;
  bool operator==(Counts const&) const = default;
};

Counts count(Storage const& s) {
  Counts c;
  for (auto const& tile : s.tiles)
    for (int lc = 0; lc < tile_cells_g; ++lc) {
      c.tile_gaps += tile.gap_idx[lc];
      for (u64 i = 0; i < tile.sizes[lc]; ++i)
        c.moved += tile.local(tile.particles[lc * cap + i].iCell_) != lc;
    }
  for (int lc = 0; lc < set_cells_g; ++lc) {
    c.set_gaps += s.pusher.set.gap_idx[lc];
    c.add_into += s.pusher.set.add_into[lc];
  }
  return c;
}

int main() {
  KOUT(NON) << __FILE__;
  Storage host{1067}, device{1067};
  host.pusher.on_host();
  device.pusher.on_device();
  auto const h = count(host), d = count(device);
  for (auto const& [name, c] : {std::pair{"host  ", h}, std::pair{"device", d}})
    std::printf("%s: moved %llu registered %llu (tile %llu + set %llu), add_into %llu\n", name,
                c.moved, c.tile_gaps + c.set_gaps, c.tile_gaps, c.set_gaps, c.add_into);
  bool const ok = h == d and d.moved == d.tile_gaps + d.set_gaps;
  std::printf("%s\n", ok ? "PASS" : "FAIL: the device lost registrations");
  return ok ? 0 : 1;
}
