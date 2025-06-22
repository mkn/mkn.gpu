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
#ifndef _MKN_GPU_MULTI_LAUNCH_HPP_
#define _MKN_GPU_MULTI_LAUNCH_HPP_

#include "mkn/gpu.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/time.hpp"
#include "mkn/kul/except.hpp"

#include <mutex>
#include <chrono>
#include <thread>
#include <vector>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace mkn::gpu::detail {
template <typename Type>
auto& deref(Type&& type) {
  if constexpr (std::is_pointer_v<std::decay_t<Type>>)
    return *type;
  else
    return type;
}
}  // namespace mkn::gpu::detail

namespace mkn::gpu {

class StreamLauncherException : public kul::Exception {
 public:
  StreamLauncherException(char const* f, std::uint16_t const& l, std::string const& s)
      : Exception{f, l, s} {}
  StreamLauncherException(StreamLauncherException const& e) : Exception{e} {}
};

enum class StreamFunctionMode { HOST_WAIT = 0, DEVICE_WAIT, BARRIER };
enum class StreamFunctionStatus { HOST_BUSY = 0, DEVICE_BUSY };

template <typename Strat>
struct StreamFunction {
  StreamFunction(Strat& strat_, StreamFunctionMode const mode_) : strat{strat_}, mode{mode_} {}
  virtual ~StreamFunction() {}
  virtual void run(std::uint32_t const) = 0;

  Strat& strat;
  StreamFunctionMode mode;
};

std::size_t inline group_idx_modulo(std::size_t const& gs, std::size_t const& i) { return i / gs; }

struct Timer {
  auto time() const {
    assert(s > 0);
    assert(e > 0);
    return e - s;
  }
  void start() { s = kul::Now::NANOS(); }
  void stop() { e = kul::Now::NANOS(); }

  std::size_t s = 0, e = 0;
};

template <typename Strat>
struct StreamGroupFunction : public StreamFunction<Strat> {
  using Super = StreamFunction<Strat>;
  StreamGroupFunction(std::size_t const gs, Strat& strat_, StreamFunctionMode const mode_)
      : Super{strat_, mode_}, group_size{gs} {}
  virtual ~StreamGroupFunction() {}

  std::size_t group_idx(std::size_t const i) const { return group_idx_modulo(group_size, i); }
  std::size_t const group_size = 0;
};

template <typename Strat, typename Fn>
struct StreamDeviceFunction : StreamFunction<Strat> {
  using Super = StreamFunction<Strat>;
  using Super::strat;

  StreamDeviceFunction(Strat& strat, Fn&& fn_)
      : Super{strat, StreamFunctionMode::DEVICE_WAIT}, fn{fn_} {}
  void run(std::uint32_t const i) override {
    //
    mkn::gpu::GDLauncher<false>{detail::deref(strat.datas[i]).size()}.stream(
        strat.streams[i], [=, fn = fn] __device__() mutable { fn(i); });
  }

  Fn fn;
};

template <typename Strat, typename Fn>
struct StreamHostFunction : StreamFunction<Strat> {
  using Super = StreamFunction<Strat>;
  StreamHostFunction(Strat& strat, Fn&& fn_)
      : Super{strat, StreamFunctionMode::HOST_WAIT}, fn{fn_} {}
  void run(std::uint32_t const i) override { fn(i); }
  Fn fn;
};

template <typename Datas, typename Self_ = void>
struct StreamLauncher {
  using This = StreamLauncher<Datas, Self_>;
  using Self = std::conditional_t<std::is_same_v<Self_, void>, This, Self_>;

  StreamLauncher(Datas& datas_) : datas{datas_}, streams(datas.size()), data_step(datas.size(), 0) {
    for (auto& s : streams) events.emplace_back(s);
  }

  ~StreamLauncher() { sync(); }

  void sync() noexcept {
    for (auto& s : streams) s.sync();
  }

  template <typename Fn>
  Self& dev(Fn&& fn) {
    fns.emplace_back(std::make_shared<StreamDeviceFunction<Self, Fn>>(self, std::forward<Fn>(fn)));
    return self;
  }
  template <typename Fn>
  Self& host(Fn&& fn) {
    fns.emplace_back(std::make_shared<StreamHostFunction<Self, Fn>>(self, std::forward<Fn>(fn)));
    return self;
  }

  void operator()() {
    using namespace std::chrono_literals;

    if (fns.size() == 0 || datas.size() == 0) return;
    times.resize(fns.size() * datas.size());

    for (std::size_t i = 0; i < datas.size(); ++i) self(i);

    do {
      for (std::size_t i = 0; i < datas.size(); ++i) {
        if (is_finished(i)) continue;
        if (is_fn_finished(i)) {
          data_step[i] += 1;
          if (not is_finished(i)) self(i);
        }
      }
      std::this_thread::sleep_for(1ms);  // make sleep time configurable
    } while (!is_finished());
  }

  void operator()(std::uint32_t const i) {
    auto const step = data_step[i];

    assert(step < fns.size());
    assert(i < events.size());

    times[(i * fns.size()) + step].start();
    fns[step]->run(i);
    if (fns[step]->mode == StreamFunctionMode::DEVICE_WAIT)
      events[i]([this, i = i]() { Self::finished_callback(this->self, i); });
  }

  void static finished_callback(This& /*self*/, std::uint32_t const& /*i*/) { /*noop*/ }

  bool is_finished() const {
    std::uint32_t finished = 0;
    for (std::size_t i = 0; i < datas.size(); ++i)
      if (is_finished(i)) ++finished;
    return finished == datas.size();
  }

  bool is_finished(std::uint32_t idx) const { return data_step[idx] == fns.size(); }

  bool is_fn_finished(std::uint32_t const& i) {
    auto const& step = data_step[i];
    auto const b = [&]() {
      if (fns[step]->mode == StreamFunctionMode::HOST_WAIT) return true;
      return events[i].finished();
    }();
    if (b) times[(i * fns.size()) + step].stop();

    return b;
  }

  Datas& datas;
  std::vector<std::shared_ptr<StreamFunction<Self>>> fns;
  std::vector<mkn::gpu::Stream> streams;
  std::vector<mkn::gpu::StreamEvent> events;
  std::vector<std::uint16_t> data_step;
  std::vector<Timer> times;
  Self& self = *reinterpret_cast<Self*>(this);
};

enum class SFS : std::uint16_t { FIRST = 0, BUSY, WAIT, FIN };
enum class SFP : std::uint16_t { WORK = 0, NEXT, SKIP };

template <typename Strat, typename Fn>
struct AsyncStreamHostFunction : StreamFunction<Strat> {
  using Super = StreamFunction<Strat>;
  using Super::strat;
  AsyncStreamHostFunction(Strat& strat, Fn&& fn_)
      : Super{strat, StreamFunctionMode::HOST_WAIT}, fn{fn_} {}
  void run(std::uint32_t const i) override {
    fn(i);
    strat.status[i] = SFS::WAIT;
  }
  Fn fn;
};

template <typename Strat>
struct StreamBarrierFunction : StreamFunction<Strat> {
  using Super = StreamFunction<Strat>;
  using Super::strat;

  StreamBarrierFunction(Strat& strat)
      : Super{strat, StreamFunctionMode::BARRIER},
        sync_point{std::ssize(strat.datas), on_completion} {}

  void run(std::uint32_t const /*i*/) override { [[maybe_unused]] auto ret = sync_point.arrive(); }

  std::function<void()> on_completion = [&]() {
    for (auto& stat : strat.status) stat = SFS::WAIT;
  };

  std::barrier<decltype(on_completion)> sync_point;
};

template <typename Strat>
struct StreamGroupBarrierFunction : StreamGroupFunction<Strat> {
  using This = StreamGroupBarrierFunction<Strat>;
  using Super = StreamGroupFunction<Strat>;
  using Super::strat;

  std::string_view constexpr static MOD_GROUP_ERROR =
      "mkn.gpu error: StreamGroupBarrierFunction Group size must be a divisor of datas";

  struct GroupBarrier {
    This* self;
    std::uint16_t group_id;

    std::function<void()> on_completion = [this]() {
      std::size_t const offset = self->group_size * group_id;
      for (std::size_t i = offset; i < offset + self->group_size; ++i)
        self->strat.status[i] = SFS::WAIT;
    };

    std::barrier<decltype(on_completion)> sync_point{static_cast<std::int64_t>(self->group_size),
                                                     on_completion};

    GroupBarrier(This& slf, std::uint16_t const gid) : self{&slf}, group_id{gid} {}
    GroupBarrier(GroupBarrier const&) = delete;
    void arrive() { [[maybe_unused]] auto ret = sync_point.arrive(); }
  };

  static auto make_sync_points(This& self, Strat const& strat, std::size_t const group_size) {
    if (strat.datas.size() % group_size > 0) throw std::runtime_error(std::string{MOD_GROUP_ERROR});
    std::vector<std::unique_ptr<GroupBarrier>> v;
    std::uint16_t const groups = strat.datas.size() / group_size;
    v.reserve(groups);
    for (std::size_t i = 0; i < groups; ++i)
      v.emplace_back(std::make_unique<GroupBarrier>(self, i));
    return v;
  }

  StreamGroupBarrierFunction(std::size_t const gs, Strat& strat)
      : Super{gs, strat, StreamFunctionMode::BARRIER},
        sync_points{make_sync_points(*this, strat, gs)} {}

  void run(std::uint32_t const i) override { sync_points[Super::group_idx(i)]->arrive(); }

  std::vector<std::unique_ptr<GroupBarrier>> sync_points;
};

template <typename Strat, typename Fn>
struct StreamHostGroupMutexFunction : StreamGroupFunction<Strat> {
  using Super = StreamGroupFunction<Strat>;
  using Super::strat;

  std::string_view constexpr static MOD_GROUP_ERROR =
      "mkn.gpu error: StreamHostGroupMutexFunction Group size must be a divisor of datas";

  static auto make_mutices(Strat const& strat, std::size_t const& group_size) {
    if (group_size == 0 || strat.datas.size() % group_size > 0)
      throw std::runtime_error(std::string{MOD_GROUP_ERROR});
    std::uint16_t const groups = strat.datas.size() / group_size;
    return std::vector<std::mutex>{groups};
  }

  StreamHostGroupMutexFunction(std::size_t const gs, Strat& strat, Fn&& fn_)
      : Super{gs, strat, StreamFunctionMode::HOST_WAIT},
        fn{fn_},
        mutices{make_mutices(strat, gs)} {}

  void run(std::uint32_t const i) override {
    std::unique_lock<std::mutex> lock(mutices[Super::group_idx(i)], std::defer_lock);

    if (lock.try_lock()) {
      fn(i);
      strat.status[i] = SFS::WAIT;  // done
    } else {
      strat.status[i] = SFS::FIRST;  // retry
    }
  }

  Fn fn;
  std::vector<std::mutex> mutices;
};

template <typename Strat, typename Fn>
struct StreamHostGroupIndexFunction : StreamGroupFunction<Strat> {
  using Super = StreamGroupFunction<Strat>;
  using Super::strat;

  std::string_view constexpr static MOD_GROUP_ERROR =
      "mkn.gpu error: StreamHostGroupIndexFunction Group size must be a divisor of datas";

  StreamHostGroupIndexFunction(std::size_t const& gs, std::size_t const& gid_, Strat& strat,
                               Fn&& fn_)
      : Super{gs, strat, StreamFunctionMode::HOST_WAIT}, fn{fn_}, gid{gid_} {
    if (gs == 0) throw std::invalid_argument("Group size cannot be zero");
  }

  void run(std::uint32_t const i) override {
    if (i % Super::group_size == gid) fn(i);
    strat.status[i] = SFS::WAIT;  // done
  }

  Fn fn;
  std::size_t const gid;
};

template <typename Strat, typename Fn, bool is = true>
struct StreamDeviceGroupIndexFunction : StreamGroupFunction<Strat> {
  using Super = StreamGroupFunction<Strat>;
  using Super::strat;

  std::string_view constexpr static MOD_GROUP_ERROR =
      "mkn.gpu error: StreamDeviceGroupIndexFunction Group size must be a divisor of datas";

  StreamDeviceGroupIndexFunction(std::size_t const gs, std::size_t const gid_, Strat& strat,
                                 Fn&& fn_)
      : Super{gs, strat, StreamFunctionMode::DEVICE_WAIT}, fn{fn_}, gid{gid_} {}

  void run(std::uint32_t const i) override {
    std::size_t const size = detail::deref(strat.datas[i]).size();

    if constexpr (is) {
      if (i % Super::group_size == gid and size)
        mkn::gpu::GDLauncher<false>{size}.stream(strat.streams[i],
                                                 [=, fn = fn] __device__() mutable { fn(i); });

    } else {
      if (i % Super::group_size != gid and size)
        mkn::gpu::GDLauncher<false>{size}.stream(strat.streams[i],
                                                 [=, fn = fn] __device__() mutable { fn(i); });
    }
  }

  Fn fn;
  std::size_t const gid;
};

template <typename Datas>
struct ThreadedStreamLauncher : public StreamLauncher<Datas, ThreadedStreamLauncher<Datas>> {
  using This = ThreadedStreamLauncher<Datas>;
  using Super = StreamLauncher<Datas, This>;
  using Super::datas;
  using Super::events;
  using Super::fns;

  constexpr static std::size_t wait_ms = _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_;
  constexpr static std::size_t wait_add_ms = _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_ADD_;
  constexpr static std::size_t wait_max_ms = _MKN_GPU_THREADED_STREAM_LAUNCHER_WAIT_MS_MAX_;

  ThreadedStreamLauncher(Datas& datas, std::size_t const _n_threads = 0,
                         std::size_t const device = 0)
      : Super{datas}, n_threads{_n_threads}, device_id{device} {
    thread_status.resize(_n_threads + 1, SFP::NEXT);
    status.resize(datas.size(), SFS::FIRST);
  }

  ~ThreadedStreamLauncher() { join(); }

  template <typename Fn>
  This& host(Fn&& fn) {
    fns.emplace_back(
        std::make_shared<AsyncStreamHostFunction<This, Fn>>(*this, std::forward<Fn>(fn)));
    return *this;
  }

  This& barrier() {
    fns.emplace_back(std::make_shared<StreamBarrierFunction<This>>(*this));
    return *this;
  }

  This& group_barrier(std::size_t const group_size) {
    fns.emplace_back(std::make_shared<StreamGroupBarrierFunction<This>>(group_size, *this));
    return *this;
  }

  template <typename Fn>
  This& host_group_mutex(std::size_t const group_size, Fn&& fn) {
    fns.emplace_back(std::make_shared<StreamHostGroupMutexFunction<This, Fn>>(
        group_size, *this, std::forward<Fn>(fn)));
    return *this;
  }

  template <typename Fn>
  This& host_group_idx(std::size_t const group_size, std::size_t const group_idx, Fn&& fn) {
    fns.emplace_back(std::make_shared<StreamHostGroupIndexFunction<This, Fn>>(
        group_size, group_idx, *this, std::forward<Fn>(fn)));
    return *this;
  }

  void static finished_callback(This& self, std::uint32_t const i) { self.status[i] = SFS::WAIT; }

  auto& operator()() { return join(); }
  Super& super() { return *this; }
  Super const& super() const { return *this; }
  void super(std::size_t const idx) { return super()(idx); }

  bool is_fn_finished(std::uint32_t const i) {
    auto const b = status[i] == SFS::WAIT;
    if (b) super().times[(i * fns.size()) + step[i]].stop();
    return b;
  }

  void thread_fn(std::size_t const /*tid*/) {
    mkn::gpu::setDevice(device_id);
    std::size_t waitms = wait_ms;
    while (!done) {
      auto const& [ts, idx] = get_work();

      if (ts == SFP::WORK) {
        waitms = wait_ms;
        super(idx);
        continue;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(waitms));
      waitms = waitms >= wait_max_ms ? wait_max_ms : waitms + wait_add_ms;
    }
  }

  bool check_finished() {
    for (std::size_t i = 0; i < datas.size(); ++i)
      if (status[i] != SFS::FIN) return false;
    return true;
  }

  std::pair<SFP, std::size_t> get_work() {
    std::unique_lock<std::mutex> lock(work_, std::defer_lock);

    if (not lock.try_lock()) return std::make_pair(SFP::SKIP, 0);

    for (; work_i < datas.size(); ++work_i) {
      auto const i = work_i;
      if (status[i] == SFS::FIN || status[i] == SFS::BUSY) continue;

      if (status[i] == SFS::FIRST) {
        status[i] = SFS::BUSY;
        return std::make_pair(SFP::WORK, i);
      }

      if (!is_fn_finished(i)) continue;

      if (status[i] == SFS::WAIT) {
        ++step[i];

        if (Super::is_finished(i)) {
          status[i] = SFS::FIN;
          continue;
        }

        status[i] = SFS::BUSY;
        return std::make_pair(SFP::WORK, i);
      }
    }
    work_i = 0;
    if (check_finished()) done = 1;
    return std::make_pair(SFP::SKIP, 0);
  }

  This& join(bool const work = true, bool const clear = false) {
    if (n_threads == 0 and !work) {
      KEXCEPT(StreamLauncherException, "no available threads, join must work");
    }
    if (!started) start();
    if (joined) return *this;
    joined = true;

    if (work) thread_fn(threads.size());
    for (auto& t : threads) t.join();
    if (clear) threads.clear();
    return *this;
  }

  This& start() {
    if (started) return *this;
    started = 1;

    Super::times.resize(fns.size() * datas.size());

    for (std::size_t i = 0; i < n_threads; ++i)
      threads.emplace_back([&, i = i]() { thread_fn(i); });
    return *this;
  }

  template <bool nl = false, typename SS>  // use div = 1e6 for milliseconds
  void _print_times(SS&& ss, double const div = 1) const {
    std::size_t fn_idx = 0, data_idx = 0;

    for (auto const& t : super().times) {
      ss << data_idx << " " << fn_idx << " " << static_cast<std::size_t>(t.time() / div);
      if constexpr (nl) ss << std::endl;
      ++fn_idx;
      if (fn_idx == fns.size()) {
        ++data_idx;
        fn_idx = 0;
      }
    }
  }

  void print_times(double const div = 1) const { _print_times(KOUT(NON), div); }
  void dump_times(std::string const& filename, double const div = 1) const {
    _print_times<1>(std::ofstream{filename}, div);
  }

  std::size_t const n_threads = 1;
  std::size_t const device_id = 0;
  std::vector<std::thread> threads;

  std::mutex work_;
  std::vector<SFS> status;
  std::vector<SFP> thread_status;
  std::vector<std::uint16_t>& step = Super::data_step;

 private:
  bool joined = false, started = false, done = false;
  std::size_t work_i = 0;
};

}  // namespace mkn::gpu

#endif /* _MKN_GPU_MULTI_LAUNCH_HPP_ */
