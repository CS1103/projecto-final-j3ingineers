cmake_minimum_required(VERSION 3.17)
project(Pong_AI)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(include/utec)
include_directories(include/utec/agent)
include_directories(include/utec/algebra)
include_directories(include/utec/nn)

set(SOURCES_COMUNES
        include/utec/thread/ConcurrentQueue.h
        include/utec/thread/ParallelExecutor.h
        include/utec/thread/ThreadPool.h
        include/utec/agent/PongAgent.h
        include/utec/agent/EnvGym.h
        include/utec/agent/State.h
        include/utec/agent/PongAgentTrainable.h
        include/utec/algebra/tensor.h
        include/utec/nn/neural_network.h
        include/utec/nn/nn_activation.h
        include/utec/nn/nn_dense.h
        include/utec/nn/nn_interfaces.h
        include/utec/nn/nn_loss.h
        include/utec/nn/nn_optimizer.h
        src/utec/agent/PongAgent.cpp
        src/utec/agent/EnvGym.cpp
        )

add_executable(Pong_AI
        ${SOURCES_COMUNES}
        main.cpp
        )

add_executable(TestPong
        ${SOURCES_COMUNES}
        tests/test_agent_env.cpp
        )
