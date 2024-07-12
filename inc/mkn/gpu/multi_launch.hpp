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
#include <thread>
#include <algorithm>

#include "mkn/gpu.hpp"

namespace mkn::gpu {

enum class StreamFunctionMode { HOST_WAIT = 0, DEVICE_WAIT };

template <typename Strat>
struct StreamFunction {
  StreamFunction(Strat& strat_, StreamFunctionMode mode_) : strat{strat_}, mode{mode_} {}
  virtual ~StreamFunction() {}
  virtual void run(std::uint32_t const) {};

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

template <typename Datas>
struct StreamLauncher {
  using This = StreamLauncher<Datas>;
  using T = typename Datas::value_type::value_type;

  StreamLauncher(Datas& datas_) : datas{datas_}, streams(datas.size()), data_step(datas.size(), 0) {
    for (auto& s : streams) events.emplace_back(s);
  }

  ~StreamLauncher() { sync(); }

  void sync() noexcept {
    for (auto& s : streams) s.sync();
  }

  template <typename Fn>
  auto& dev(Fn&& fn) {
    fns.emplace_back(std::make_shared<StreamDeviceFunction<This, Fn>>(self, std::forward<Fn>(fn)));
    return self;
  }
  template <typename Fn>
  auto& host(Fn&& fn) {
    fns.emplace_back(std::make_shared<StreamHostFunction<This, Fn>>(self, std::forward<Fn>(fn)));
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
    fns[step]->run(i);
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
    if (b) events[i].reset();
    return b;
  }

  Datas& datas;
  std::vector<std::shared_ptr<StreamFunction<This>>> fns;
  std::vector<mkn::gpu::Stream> streams;
  std::vector<mkn::gpu::StreamEvent> events;
  std::vector<std::uint16_t> data_step;
  This& self = *this;
};

}  // namespace mkn::gpu

#endif /* _MKN_GPU_MULTI_LAUNCH_HPP_ */
