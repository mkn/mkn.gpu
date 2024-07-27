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

#include <cassert>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>

#include "mkn/gpu.hpp"

namespace mkn::gpu {

enum class StreamFunctionMode { HOST_WAIT = 0, DEVICE_WAIT };
enum class StreamFunctionStatus { HOST_BUSY = 0, DEVICE_BUSY };

template <typename Strat>
struct StreamFunction {
  StreamFunction(Strat& strat_, StreamFunctionMode mode_) : strat{strat_}, mode{mode_} {}
  virtual ~StreamFunction() {}
  virtual void run(std::uint32_t const) = 0;

  Strat& strat;
  StreamFunctionMode mode;
};

template <typename Strat, typename Fn>
struct StreamDeviceFunction : StreamFunction<Strat> {
  using Super = StreamFunction<Strat>;
  using Super::strat;

  StreamDeviceFunction(Strat& strat, Fn&& fn_)
      : Super{strat, StreamFunctionMode::DEVICE_WAIT}, fn{fn_} {}
  void run(std::uint32_t const i) override {
    strat.events[i].record();

    mkn::gpu::GDLauncher<false>{strat.datas[i].size()}.stream(
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

    if (fns.size() == 0) return;

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
    auto const& step = data_step[i];
    assert(step < fns.size());
    fns[step]->run(i);
    assert(i < events.size());
    assert(step < fns.size());
    if (fns[step]->mode == StreamFunctionMode::DEVICE_WAIT) events[i].record();
  }

  bool is_finished() const {
    std::uint32_t finished = 0;
    for (std::size_t i = 0; i < datas.size(); ++i)
      if (is_finished(i)) ++finished;
    return finished == datas.size();
  }

  bool is_finished(std::uint32_t idx) const { return data_step[idx] == fns.size(); }

  bool is_fn_finished(std::uint32_t i) {
    auto const b = [&]() {
      auto const& step = data_step[i];
      if (fns[step]->mode == StreamFunctionMode::HOST_WAIT) return true;
      return events[i].finished();
    }();
    if (b) {
      events[i].reset();
    }
    return b;
  }

  Datas& datas;
  std::vector<std::shared_ptr<StreamFunction<Self>>> fns;
  std::vector<mkn::gpu::Stream> streams;
  std::vector<mkn::gpu::StreamEvent> events;
  std::vector<std::uint16_t> data_step;
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

template <typename Datas>
struct ThreadedStreamLauncher : public StreamLauncher<Datas, ThreadedStreamLauncher<Datas>> {
  using This = ThreadedStreamLauncher<Datas>;
  using Super = StreamLauncher<Datas, This>;
  using Super::datas;
  using Super::events;
  using Super::fns;

  constexpr static std::size_t wait_ms = 1;
  constexpr static std::size_t wait_max_ms = 100;

  ThreadedStreamLauncher(Datas& datas, std::size_t const _n_threads = 1)
      : Super{datas}, n_threads{_n_threads} {
    thread_status.resize(n_threads, SFP::NEXT);
    status.resize(datas.size(), SFS::FIRST);
  }

  ~ThreadedStreamLauncher() { join(); }

  template <typename Fn>
  This& host(Fn&& fn) {
    fns.emplace_back(
        std::make_shared<AsyncStreamHostFunction<This, Fn>>(*this, std::forward<Fn>(fn)));
    return *this;
  }

  void operator()() { join(); }
  Super& super() { return *this; }
  void super(std::size_t const& idx) { return super()(idx); }

  bool is_fn_finished(std::uint32_t i) {
    auto const b = [&]() {
      if (fns[step[i]]->mode == StreamFunctionMode::HOST_WAIT) return status[i] == SFS::WAIT;
      return events[i].finished();
    }();
    if (b) {
      events[i].reset();
      status[i] = SFS::WAIT;
    }
    return b;
  }
  void thread_fn(std::size_t const& /*tid*/) {
    cudaSetDevice(0);  // configurable
    std::size_t waitms = wait_ms;
    while (!done) {
      auto const& [ts, idx] = get_work();

      if (ts == SFP::WORK) {
        waitms = wait_ms;
        super(idx);

      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(waitms));
        waitms = waitms >= wait_max_ms ? wait_max_ms : waitms + 10;
        if (check_finished()) done = 1;
      }
    }
  }

  bool check_finished() {
    for (std::size_t i = 0; i < datas.size(); ++i)
      if (status[i] != SFS::FIN) return false;
    return true;
  }

  std::pair<SFP, std::size_t> get_work(std::size_t const& start = 0) {
    std::unique_lock<std::mutex> lk(work_);
    for (std::size_t i = start; i < datas.size(); ++i) {
      if (status[i] == SFS::BUSY) {
        if (is_fn_finished(i)) status[i] = SFS::WAIT;

      } else if (status[i] == SFS::WAIT) {
        ++step[i];

        if (Super::is_finished(i)) {
          status[i] = SFS::FIN;
          continue;
        }

        status[i] = SFS::BUSY;
        return std::make_pair(SFP::WORK, i);

      } else if (status[i] == SFS::FIRST) {
        status[i] = SFS::BUSY;
        return std::make_pair(SFP::WORK, i);
      }
    }

    return std::make_pair(SFP::SKIP, 0);
  }

  This& join(bool const& clear = false) {
    if (!started) start();
    if (joined) return *this;
    joined = true;

    for (auto& t : threads) t.join();
    if (clear) threads.clear();
    return *this;
  }

  This& start() {
    if (started) return *this;
    started = 1;
    for (std::size_t i = 0; i < n_threads; ++i)
      threads.emplace_back([&, i = i]() { thread_fn(i); });
    return *this;
  }

  std::size_t const n_threads = 1;
  std::vector<std::thread> threads;

  std::mutex work_;
  std::vector<SFS> status;
  std::vector<SFP> thread_status;
  std::vector<std::uint16_t>& step = Super::data_step;

 private:
  bool joined = false, started = false, done = false;
};

}  // namespace mkn::gpu

#endif /* _MKN_GPU_MULTI_LAUNCH_HPP_ */
