# Reinforcement-Learning-for-Aircraft-Control-and-Solving-Faults-Scenarios
甚至航空业的事故率在2008年和2019年之间也下降了36%。然而，飞行中失去
控制仍然是导致2009年至2018年61%的飞行事故的主要原因[1]。
因此，人们更加关注如何在一些变化的条件或极端紧急情况下保持无人驾驶飞行器（UAV）的稳定
在一些变化的条件下或极端的紧急情况下保持稳定。
此外，当环境或飞机状况发生变化时，飞机控制模型的某些指标不容易调整。
飞机的状况发生变化时，飞机控制模型的一些指标不容易调整。
一些深度强化学习方法，如软演员批评法（SAC）、信任区域政策
优化（TRPO）、近端策略优化（PPO）、深度确定型策略梯度（DDPG）、深度Quality优化（DDPG）等。
(DDPG)、深度Q网络(DQN)、世界模型和AlphaZero已经在我们的生活中得到了应用。
[2].
在这项研究中，该项目旨在应用强化学习来建立和调整飞机的动态
在这项研究中，项目旨在应用强化学习来建立和调整飞机的动态模型，以解决四旋翼飞机中的一个旋翼有随机输出率的问题。
输出。人们对无人机在发生紧急情况时应对问题的能力越来越感兴趣。
紧急情况，如高风速、副翼断裂或四旋翼飞机的发动机故障。如何
如何在紧急情况下保持无人机的稳定，以及如何控制其参数，如调整旋翼，以保持其速度。
转子的速度以保持稳定，对我们来说仍然是一个挑战。
首先，会有一个四旋翼飞机的飞行动力学模型内置模拟。仿真的内容包括
无人机的动力学和理论将被建模。在这个模型中，将有无人机的尺寸
无人机的尺寸，重量，以及无人机飞行时的风阻系数等。在这个模型中。
将有4个输入，包括无人机的螺旋桨的输出功率，和12个输出。
包括关于无人机的速度和位置的信息。
7
其次，该模型将作为旋翼的正常输入进行测试，其结果如位置和速度信息将被观察到。
同时观察位置和速度信息。无人机螺旋桨的输出功率有4个输入
观察无人机螺旋桨的输出功率的4个输入，不同的数据组合将被测试，以观察12个输出，包括位置和速度的连续变化。
输出，包括无人机的位置和速度的连续变化。
第三，无人机有一些基本的运动模式，包括悬停，即在空中稳定下来。
空中，上升，下降，左，右，前后移动，以及旋转。无人机的运动
无人机的运动是通过控制四个螺旋桨的速度来控制的，这就是四个螺旋桨的扭矩输出。
四个螺旋桨的扭矩输出。我们传统上使用一个PID控制器来控制无人机的基本运动。这项
研究试图使用强化学习来创建运动模型，而不是传统的
PID控制器。这些模型被反复训练，并增加训练次数
以检查模型是否符合要求。
最后，飞行故障将通过修改模拟器和强化学习来建立模型。
将有助于实现对故障的稳健性。实验的范围有很多，如
转子的不同类型的故障及其相应的处理方法。强化学习的模型
强化学习模型可以帮助飞机自动学习调整其参数以判断
环境并适应新的环境，以保持无人机的稳定性。

Reinforcement Learning for Aircraft Control and Solving Faults Scenarios
Even the aviation industry’s accident rate fell 36% between 2008 and 2019. However, In-flight loss
of control is still the main reason leading to the 61% of flight accidents from 2009 to 2018 [1].
Therefore more attention has been paid to how to remain Unmanned Aerial Vehicles (UAV) stable
in some changing conditions or extreme emergencies.
In addition, some index of the aircraft control model is not easy to adjust when the environment or
the condition of the aircraft change.
Some deep reinforcement learning methods like Soft Actor-Critic (SAC), Trust Region Policy
Optimization (TRPO), Proximal Policy Optimization (PPO), Deep Deterministic Policy Gradient
(DDPG), Deep Q-Networks (DQN), World Models, and AlphaZero have been applied in our lives
[2].
In this research, the project aims to apply reinforcement learning to build and adjust the dynamic
model of aircraft to solve the problem that one of the rotors of the quadrotor has a rate of random
output. Here is growing interest in the ability of drones to cope with problems in the event of an
emergency, such as high wind speed, a broken aileron ,or an engine failure on a quadcopter. How
to keep the UAV stable in an emergency, and how to control its parameters, such as rotors to adjust
the speed of the rotors to maintain stability, are still a challenge to us.
In the first place, there will be a flight dynamics model of a quadrotor built-in simulation. The
dynamics and theory of the drone will be modeled. In this heel model, there will be the dimensions
of the drone, the weight, and the wind resistance coefficient of the drone in flight, etc. In this model,
there will be 4 inputs, including the output power of the propellers of the drone, and 12 outputs,
including information about the speed and position of the drone.
7
Secondly, the model will be tested as the normal input of the rotors and the results like the position
and velocity information will be observed at the same time. There are 4 inputs of the output power
of the propeller of the drone, different kinds of data combinations will be tested to see the 12
outputs, including the continuous change of position and speed of the drone.
Thirdly, the drone has some basic movement modes, including hovering i.e. stabilization in the
air, ascent, and descent, left, right, back and forth movement, and rotation. The motion of the
UAV is controlled by controlling the speed of the four propellers, which is the torque output of the
four propellers. We traditionally use a PID controller to control the basic motion of the UAV. This
research has tried to use reinforcement learning to create the motion models instead of a traditional
PID controller. The models are trained repeatedly, and the number of training sessions is increased
to check if the models meet the requirements.
Finally, flight faults will be modeled by modifying the simulator and the reinforcement learning
will contribute to being robust to the fault. There are many scopes of experimentation, like the
different types of faults of the rotor and their corresponding processing methods. The model of
reinforcement learning can help the aircraft automatically learn to adjust their parameters to judge
the environment and adapt to the new environment to remain the drone stable
