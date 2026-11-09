/**
Copyright (c) 2026, Philip Deegan.
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
#ifndef MKN_GPU_GRAPH_HPP
#define MKN_GPU_GRAPH_HPP

#include "mkn/gpu.hpp"
#include "mkn/gpu/trace.hpp"
#include "mkn/gpu/stream.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace mkn::gpu {

// steps per data item as StreamLauncher, captured into one Graph on the first call and replayed
//  by every call after, one launch for all items and steps. Items are parallel branches on their
//  own streams, each item's steps run in order. Host steps are host nodes in the graph, they run
//  in stream order and must not call the GPU API. As for any graph, kernel arguments are fixed at
//  capture, call reset() for a new capture when host values the steps read change
//  On CPU every call runs the steps. See doc/tracing/graphs.md
template <typename Datas>
struct GraphLauncher {
  using This = GraphLauncher<Datas>;

  GraphLauncher(Datas& datas_) : datas{datas_}, streams(datas.size()) {
    if constexpr (trace::enabled)
      for (std::size_t i = 0; i < streams.size(); ++i)
        trace::name_stream(streams[i], ("mkn.gpu graph stream " + std::to_string(i)).c_str());
  }

  ~GraphLauncher() { sync(); }

  template <typename Fn>
  This& dev(Fn&& fn) {
    reset();
    fns.emplace_back(std::make_shared<StreamDeviceFunction<This, Fn>>(*this, std::forward<Fn>(fn)));
    return *this;
  }

  template <typename Fn>
  This& host(Fn&& fn) {
    reset();
    fns.emplace_back(std::make_shared<StreamHostFunction<This, Fn>>(*this, std::forward<Fn>(fn)));
    return *this;
  }

  // asynchronous, sync() to wait
  This& operator()() {
    if (fns.empty() || datas.size() == 0) return *this;
    if (graph.empty()) capture();
    trace::Range range{"GraphLauncher"};
    graph.launch(streams[0]);
    return *this;
  }

  // the graph is launched on the first stream, all branches rejoin it
  void sync() noexcept {
    if (streams.size()) streams[0].sync();
  }

  void reset() { graph.reset(); }

  struct HostNode {
    StreamFunction<This>* fn;
    std::uint32_t i;
  };

  static void run_host_node(void* const ptr) {
    auto const& node = *static_cast<HostNode*>(ptr);
    node.fn->run(node.i);
  }

  void capture() {
    sync();  // a previous graph may still be running its host nodes
    nodes.clear();
    nodes.reserve(datas.size() * fns.size());  // stable, the graph keeps their addresses
    for (std::uint32_t i = 0; i < datas.size(); ++i)
      for (auto& fn : fns)
        if (fn->mode == StreamFunctionMode::HOST_WAIT) nodes.emplace_back(fn.get(), i);

    trace::Range range{"GraphLauncher::capture"};
    graph.capture(streams, [this] {
      std::size_t n = 0;
      for (std::uint32_t i = 0; i < datas.size(); ++i)
        for (auto& fn : fns) {
          if (fn->mode == StreamFunctionMode::DEVICE_WAIT)
            fn->run(i);
          else
            launch_host_fn(streams[i], &run_host_node, &nodes[n++]);
        }
    });
  }

  Datas& datas;
  std::vector<mkn::gpu::Stream> streams;
  std::vector<std::shared_ptr<StreamFunction<This>>> fns;
  std::vector<HostNode> nodes;
  Graph graph;
};

}  // namespace mkn::gpu

#endif /* MKN_GPU_GRAPH_HPP */
