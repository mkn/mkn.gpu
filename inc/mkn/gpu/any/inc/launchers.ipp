/**
Copyright (c) 2024, Philip Deegan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Philip Deegan nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef MKN_GPU_ANY_INC_LAUNCHERS_IPP
#define MKN_GPU_ANY_INC_LAUNCHERS_IPP

namespace detail {

template <std::size_t... I, typename... Args>
auto _as_values(std::tuple<Args&...>&& tup, std::index_sequence<I...>) {
  // kernel arguments are by value, const would not match the kernel's own parameter types
  using T = std::tuple<std::remove_cvref_t<decltype(MKN_GPU_NS::replace(std::get<I>(tup)))>&...>*;
  return T{nullptr};
}

template <typename... Args>
auto as_values(Args&... args) {
  return _as_values(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>());
}

// cached per thread and device, attribute queries are not free
std::size_t inline max_threads_per_block(std::size_t const dev) {
  thread_local std::vector<std::size_t> cache;
  if (cache.size() <= dev) cache.resize(dev + 1, 0);
  if (cache[dev] == 0) cache[dev] = getMaxThreadsPerBlock(dev);
  return cache[dev];
}

template <typename T>
concept span_like = mkn::kul::is_span_like_v<std::decay_t<T>>;

// launches of kernel f asking for more dynamic shared memory than it allows on the current device
//  opt it in; the kernel's own limit accounts for its static shared memory and the attribute is
//  per device, so both are kept per kernel and device. Beyond the opt-in maximum the launch
//  itself reports the error
void inline fit_shared(void const* const f, std::size_t const bytes) {
  if (bytes == 0) return;
  static std::mutex mutex;
  static std::map<std::pair<void const*, std::size_t>, std::size_t> allowed;
  std::lock_guard<std::mutex> lock{mutex};
  auto const [it, fresh] = allowed.try_emplace({f, getDevice()}, 0);
  if (fresh) it->second = getKernelMaxDynamicSharedMemory(f);
  if (bytes <= it->second) return;
  setKernelMaxDynamicSharedMemory(f, bytes);
  it->second = bytes;
}

}  // namespace detail

// most threads per block a launch of kernel can use, queried once per kernel on the device that
//  is current at the first call, the same kernel must not then be launched on a device of
//  another architecture. Callers using several architectures keep their own value per device,
//  e.g. a static std::vector of getKernelMaxThreadsPerBlock with each device set in turn
template <auto kernel>
std::size_t kernel_bx_threads() {
  static std::size_t const max = getKernelMaxThreadsPerBlock(reinterpret_cast<void const*>(kernel));
  return max;
}

// grid/block derived from a problem size, device functions are bounds checked against it
struct Guess {};

// grid/block provided by the caller
struct Fixed {};

// as kernel_bx_threads for the kernel a DLauncher generates for device function F, global_gd_kernel
//  for Guess (bounds checked) or global_d_kernel for Fixed, separate kernels whose limits can
//  differ. Keyed on types as a local lambda type in F does not make a valid template argument for
//  all compilers
template <typename Mode, typename F, typename... PArgs>
std::size_t device_kernel_bx_threads() {
  static_assert(std::is_same_v<Mode, Guess> || std::is_same_v<Mode, Fixed>,
                "Launcher mode must be Guess or Fixed");
  auto const kernel = [] {
    if constexpr (std::is_same_v<Mode, Guess>)
      return &global_gd_kernel<F, PArgs...>;
    else
      return &global_d_kernel<F, PArgs...>;
  }();
  static std::size_t const max = getKernelMaxThreadsPerBlock(reinterpret_cast<void const*>(kernel));
  return max;
}

// most threads per block for the kernel a DLauncher<Mode> generates to launch fn(args...)
//  F keeps its const as DLauncher's does, a const and a non const functor are different kernels
template <typename Mode = Guess, typename F, typename... Args>
std::size_t getDeviceKernelMaxThreadsPerBlock(F&&, Args&&... args) {
  return [&]<typename... PArgs>(std::tuple<PArgs&...>*) {
    return device_kernel_bx_threads<Mode, std::remove_reference_t<F>, PArgs...>();
  }(detail::as_values(args...));
}

// launches are asynchronous, call sync() to wait for the stream (or device without a stream)
template <typename Mode>
struct ALauncher : public Launcher {
  static constexpr bool guess = std::is_same_v<Mode, Guess>;
  static_assert(guess || std::is_same_v<Mode, Fixed>, "Launcher mode must be Guess or Fixed");

  ALauncher(std::size_t const size, std::size_t const dev = 0)
    requires(guess)
      : Launcher{dim3{}, dim3{}}, count{size} {
    bx_cap = detail::max_threads_per_block(dev);
    set_bx(bx_cap);  // the size is checked at launch, against the final block size
  }

  template <detail::span_like C>
  ALauncher(C const& c, std::size_t const dev = 0)
    requires(guess)
      : ALauncher{static_cast<std::size_t>(c.size()), dev} {}

  ALauncher()
    requires(!guess)
      : Launcher{dim3{1}, dim3{warp_size}} {}

  ALauncher(dim3 const _g, dim3 const _b)
    requires(!guess)
      : Launcher{_g, _b} {}

  ALauncher(std::size_t const w, std::size_t const h, std::size_t const tpx, std::size_t const tpy)
    requires(!guess)
      : Launcher{w, h, tpx, tpy} {}

  ALauncher(std::size_t const x, std::size_t const y, std::size_t const z, std::size_t const tpx,
            std::size_t const tpy, std::size_t const tpz)
    requires(!guess)
      : Launcher{x, y, z, tpx, tpy, tpz} {}

  auto& sync() {
    if (s)
      MKN_GPU_NS::sync(s);
    else
      MKN_GPU_NS::sync();
    return *this;
  }

  bool empty() const {
    if constexpr (guess) return count == 0;
    return false;
  }

  std::size_t count = 0;   // Guess only
  std::size_t bx_cap = 0;  // Guess only, block x dimension before the kernel limit

 protected:
  void set_bx(std::size_t const bx) {
    b.x = bx;
    g.x = mkn::gpu::guess_blocks(count, bx);
  }

  // before a guessed launch, the grid of the final block size must fit in 32 bits
  void check() {
    if constexpr (guess) g.x = mkn::gpu::guess_grid_dim(count, b.x);
  }

  // guessed launches are limited by the kernel's own maximum block size, which can be below the
  //  device maximum for kernels with high register or shared memory use
  void fit(std::size_t const kernel_max) {
    if constexpr (guess) set_bx(mkn::gpu::clamp_bx_threads(bx_cap, kernel_max, warp_size));
  }
};

// for __global__ functions
template <typename Mode>
struct GlobalLauncher : public ALauncher<Mode> {
  using Super = ALauncher<Mode>;
  using Super::Super;

  // guessed launches of __global__ functions are not limited by the kernel unless given its
  //  maximum, e.g. once per kernel: static auto const bx = kernel_bx_threads<kernel>();
  auto& bx(std::size_t const kernel_max)
    requires(Super::guess)
  {
    this->fit(kernel_max);
    return *this;
  }

  template <typename F, typename... Args>
  auto& operator()(F&& f, Args&&... args) {
    if (this->empty()) return *this;
    this->check();
    detail::fit_shared(reinterpret_cast<void const*>(+f), this->ds);
    MKN_GPU_NS::launch<false>(std::forward<F>(f), this->g, this->b, this->ds, this->s, args...);
    return *this;
  }

  template <typename F, typename... Args>
  auto& stream(Stream& _s, F&& f, Args&&... args) {
    this->s = _s();
    return (*this)(std::forward<F>(f), args...);
  }
};

// for __device__ functions/lambdas
template <typename Mode>
struct DeviceLauncher : public ALauncher<Mode> {
  using Super = ALauncher<Mode>;
  using Super::Super;

  template <typename F, typename... Args>
  auto& operator()(F&& f, Args&&... args) {
    if (this->empty()) return *this;
    _launch(f, detail::as_values(args...), args...);
    return *this;
  }

  template <typename F, typename... Args>
  auto& stream(Stream& _s, F&& f, Args&&... args) {
    this->s = _s();
    return (*this)(std::forward<F>(f), args...);
  }

 protected:
  template <typename F, typename... PArgs, typename... Args>
  void _launch(F& f, std::tuple<PArgs&...>*, Args&&... args) {
    if constexpr (Super::guess) {
      this->fit(device_kernel_bx_threads<Guess, F, PArgs...>());
      this->check();
      detail::fit_shared(reinterpret_cast<void const*>(&global_gd_kernel<F, PArgs...>), this->ds);
      MKN_GPU_NS::launch<false>(&global_gd_kernel<F, PArgs...>, this->g, this->b, this->ds, this->s,
                                f, this->count, args...);
    } else {
      detail::fit_shared(reinterpret_cast<void const*>(&global_d_kernel<F, PArgs...>), this->ds);
      MKN_GPU_NS::launch<false>(&global_d_kernel<F, PArgs...>, this->g, this->b, this->ds, this->s,
                                f, args...);
    }
  }
};

// short names are class templates rather than alias templates, not all compilers can deduce
//  through an alias yet (clang < 19)
template <typename Mode>
struct GLauncher : public GlobalLauncher<Mode> {
  using GlobalLauncher<Mode>::GlobalLauncher;
};

template <typename Mode>
struct DLauncher : public DeviceLauncher<Mode> {
  using DeviceLauncher<Mode>::DeviceLauncher;
};

#define MKN_GPU_LAUNCHER_DEDUCTION_GUIDES(L)                         \
  L(std::size_t) -> L<Guess>;                                        \
  L(std::size_t, std::size_t) -> L<Guess>;                           \
  template <detail::span_like C>                                     \
  L(C const&) -> L<Guess>;                                           \
  template <detail::span_like C>                                     \
  L(C const&, std::size_t) -> L<Guess>;                              \
  L() -> L<Fixed>;                                                   \
  L(dim3, dim3) -> L<Fixed>;                                         \
  L(std::size_t, std::size_t, std::size_t, std::size_t) -> L<Fixed>; \
  L(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t) -> L<Fixed>;

MKN_GPU_LAUNCHER_DEDUCTION_GUIDES(GlobalLauncher)
MKN_GPU_LAUNCHER_DEDUCTION_GUIDES(DeviceLauncher)
MKN_GPU_LAUNCHER_DEDUCTION_GUIDES(GLauncher)
MKN_GPU_LAUNCHER_DEDUCTION_GUIDES(DLauncher)

#undef MKN_GPU_LAUNCHER_DEDUCTION_GUIDES

#endif /* MKN_GPU_ANY_INC_LAUNCHERS_IPP */
