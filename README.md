# Physics-Guided Energy-Efficient Trajectory Planning for a 6-DOF Industrial Manipulator Using Soft Actor-Critic Reinforcement Learning

<p align="center">
  <img width="300" height="320" alt="6 DOF Manipulator" src="https://github.com/user-attachments/assets/4cc89cee-b8ad-4a51-a35f-7fe2339278bc" />

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Under%20Revision-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Journal-Robotics%20and%20Autonomous%20Systems-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-SAC-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Simulation-PyBullet-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Topic-Energy%20Efficient%20Robotics-yellow?style=for-the-badge"/>
</p>

---

# 📖 Project Overview

This repository accompanies the research manuscript **“Physics-Guided Energy-Efficient Trajectory Planning for a 6-DOF Industrial Manipulator Using Soft Actor-Critic Reinforcement Learning.”**

The study proposes a **Physics-Guided Deep Reinforcement Learning (PG-DRL)** framework based on **Soft Actor-Critic (SAC)** for energy-efficient motion planning of a **6-degree-of-freedom industrial robot manipulator**. The framework is designed to jointly optimize **trajectory tracking accuracy, energy consumption, and motion smoothness**.

Unlike conventional PID-style motion control, this work integrates:
- a **physics-consistent simulation environment**,
- a **temporally aware state representation** with historical energy information,
- and **automated reward-weight optimization** using Bayesian optimization.

The framework is validated in a high-fidelity **PyBullet** simulation and is intended as a foundation for **energy-efficient industrial manipulation** in coarse-positioning tasks such as material handling and large-workpiece transport.

---

# 🎯 Research Objectives

- Develop an energy-aware reinforcement learning framework for a 6-DOF industrial manipulator.
- Reduce total energy consumption during trajectory planning.
- Improve motion smoothness by lowering mechanical jerk.
- Maintain acceptable trajectory tracking performance.
- Integrate physics knowledge into the RL training loop.
- Automate reward-weight selection using Bayesian optimization.
- Provide a reproducible simulation-based control pipeline.

---

# ✨ Key Features

- Soft Actor-Critic (SAC) based control
- Physics-guided reinforcement learning
- Energy-aware state representation
- Historical energy memory (`Ehistory`)
- Direct position-based joint control
- High-fidelity PyBullet simulation
- Multi-objective reward design
- Automated reward tuning with Optuna
- Joint torque and jerk-aware optimization
- Industrial manipulator trajectory planning

---

# 🏗 System Overview

The proposed framework is built around a closed-loop RL workflow:

1. The agent observes the robot state.
2. The SAC policy outputs continuous joint position increments.
3. The PyBullet environment updates robot dynamics.
4. Reward is computed using tracking error, energy, safety, and smoothness.
5. The policy and critic networks are updated from replayed experience.

This design explicitly models robot dynamics, friction, and gravity so that the learned policy reflects physically meaningful motion behavior.

---

# ⚙️ Hardware Used

| Hardware / Platform | Why it was used |
|---|---|
| High-performance workstation | To run RL training and simulation experiments efficiently |
| GPU-accelerated computing platform | To speed up neural network optimization and parallel simulation |
| 6-DOF industrial manipulator model (URDF) | To represent a realistic industrial robot with six revolute joints |
| Simulation environment (PyBullet-based ArmReach6DOFEnv) | To model dynamics, friction, gravity, and joint constraints without physical hardware deployment |

---

# 💻 Software Used

| Software | Why it was used |
|---|---|
| Python | Main implementation language for training and evaluation |
| PyTorch | Implementation of actor, critic, and training networks |
| PyBullet | Physics simulation for robot dynamics and trajectory execution |
| Optuna | Bayesian hyperparameter optimization and reward-weight tuning |
| TPE sampler | Efficient search strategy inside Optuna |
| CUDA / GPU support | Faster deep reinforcement learning training |

---

# 🔧 Methods Used

| Method | Why it was important |
|---|---|
| Soft Actor-Critic (SAC) | Handles high-dimensional continuous control well |
| Physics-guided RL | Makes the learning process physically consistent |
| Denavit–Hartenberg (D-H) modeling | Defines robot geometry and forward kinematics |
| Homogeneous transformation matrices | Computes end-effector pose in task space |
| Euler–Lagrange dynamic model | Captures inertia, Coriolis, centrifugal, and gravity effects |
| Energy modeling from torque and velocity | Measures energy usage in a physically meaningful way |
| Temporally aware state representation | Adds long-term energy memory to the agent’s observation |
| Multi-objective reward design | Balances tracking, energy, safety, and smoothness |
| Optuna Bayesian optimization | Removes manual bias in reward-weight selection |

---

# 🛠 Technical Skills Demonstrated

This work demonstrates experience in:

- Reinforcement Learning
- Deep Reinforcement Learning
- Robotics Control
- Industrial Manipulator Modeling
- Physics-Based Simulation
- PyBullet Simulation
- PyTorch Implementation
- Energy-Aware Control Design
- Multi-Objective Optimization
- Bayesian Hyperparameter Tuning
- Robotics Kinematics and Dynamics
- Sustainable Industrial Automation

---

# 📌 Intended Use

This repository is intended to support:
- academic review,
- reproducibility of the simulation pipeline,
- research presentation,
- and future hardware deployment work.

---

---

# 📊 Results & Performance Evaluation

The proposed **Physics-Guided Soft Actor-Critic (PG-SAC)** framework was evaluated in the ArmReach6DOFEnv simulation using PyBullet and compared with a conventional PID controller and a standard SAC baseline.

## 🏆 Quantitative Performance Comparison

| 📈 Performance Metric | PID Controller | Standard SAC | Proposed PG-SAC |
|:---|---:|---:|---:|
| Mean Tracking Error (mm) | 450 | 380 | **211** |
| Total Energy Consumption (J) | 29.5 | 23.0 | **17.5** |
| Maximum Jerk (rad/s³) | 25,000 | 22,000 | **17,800** |
| Task Success Rate (%) | 65% | 75% | **85%** |

## 🔑 Key Outcomes

- ⚡ **Energy Efficiency:** Approximately 40% lower energy consumption than the PID baseline.
- 🎯 **Tracking Performance:** Approximately 53% lower mean tracking error than PID.
- 🔄 **Motion Smoothness:** A 28.8% reduction in maximum mechanical jerk compared with PID.
- 🤖 **Task Success:** An 85% task success rate, compared with 65% for PID and 75% for standard SAC.
- 🧠 **Learning Stability:** Reward convergence over 1,500 training episodes, with improved energy-aware control behavior.
- 🌐 **Multi-Target Performance:** The learned policy was also evaluated on sequential target-tracking tasks with changing end-effector goals.

These results demonstrate the potential of combining physics-guided learning, historical energy information, and multi-objective reward optimization for energy-efficient robotic motion planning.

---

# 🔬 Main Research Contributions

The principal contributions of this work are:

| Contribution | Technical Significance |
|:---|:---|
| Physics-Guided SAC Framework | Integrates physical robot dynamics into reinforcement learning for energy-aware control. |
| Temporal Energy Memory (`Ehistory`) | Incorporates historical energy consumption into the agent's state representation. |
| Multi-Objective Reward Function | Balances tracking error, energy expenditure, safety constraints, and motion smoothness. |
| Physics-Based Energy Estimation | Uses joint torque, angular velocity, and actuator-loss approximations to estimate energy expenditure. |
| Bayesian Reward Optimization | Uses Optuna with a Tree-structured Parzen Estimator (TPE) to optimize reward weights. |
| Comparative Evaluation | Benchmarks PG-SAC against PID and standard SAC using tracking error, energy consumption, jerk, and task success rate. |

---

# 🧠 Why Physics-Guided Reinforcement Learning?

Conventional PID control can provide effective feedback control but may generate inefficient motion profiles when robot dynamics and energy consumption are not explicitly considered.

The proposed PG-SAC framework addresses this challenge through three integrated mechanisms:

- **Physics-aware learning:** The simulation models robot dynamics, gravitational effects, friction, and joint constraints.
- **Energy-aware decision-making:** Historical energy information helps the agent account for the longer-term consequences of its actions.
- **Smoothness-aware optimization:** The reward design penalizes undesirable motion and mechanical jerk while encouraging tracking progress and energy conservation.

This integrated approach supports the development of more sustainable robotic control strategies aligned with **Industry 5.0** objectives.

---

# ⚠️ Scope, Limitations & Responsible Interpretation

The reported results are obtained from **simulation experiments in PyBullet**. They should not be interpreted as proof of equivalent performance on a physical industrial robot.

Important limitations include:

- **Simulation-only validation:** Hardware deployment and sim-to-real transfer have not yet been demonstrated.
- **Tracking accuracy:** The reported mean tracking error of 211 mm remains too high for precision operations such as welding and other millimetre-level tasks.
- **Robot generalization:** The study evaluates a single KUKA KR-class 6-DOF manipulator model.
- **Baseline sensitivity:** PID performance depends on controller tuning, and further comparisons with carefully tuned conventional controllers would strengthen the evaluation.
- **Statistical validation:** Additional experiments across independent random seeds and robotic configurations are needed to establish broader generalizability.

The present framework is therefore best viewed as a **simulation-validated foundation for energy-aware coarse-positioning tasks**, rather than a deployment-ready precision controller.

---

# 🚀 Future Research Directions

Future development will focus on:

- 🔧 **Sim-to-Real Transfer:** Deploying the learned policy on a physical manipulator and addressing simulation-to-hardware discrepancies.
- 🎯 **Improved Tracking Accuracy:** Refining reward design and control strategies toward sub-50 mm tolerances.
- 🤖 **Cross-Robot Generalization:** Evaluating the framework on different 6-DOF industrial manipulators.
- 📈 **Robustness Testing:** Repeating experiments across multiple random seeds, target configurations, and operating conditions.
- ⚙️ **Advanced Energy Modeling:** Incorporating more detailed actuator and electrical-loss models.
- 🧪 **Expanded Benchmarking:** Comparing against additional reinforcement learning algorithms and carefully tuned model-based controllers.
- 🛡️ **Constraint-Aware Control:** Strengthening the handling of joint limits, collisions, and operational safety constraints.

---

# 🌍 Potential Industrial Applications

The current simulation-based framework is most relevant to applications where energy efficiency and smooth motion are important and millimetre-level precision is not the primary requirement.

| Application | Potential Relevance |
|:---|:---|
| Bulk Material Handling | Energy-aware movement of materials between locations. |
| Large-Workpiece Transport | Efficient repositioning of heavy or bulky industrial components. |
| Industrial Pre-Positioning | Moving robotic arms toward target regions before fine positioning. |
| Automated Material Transfer | Reducing unnecessary motion and energy expenditure during repetitive transfers. |
| Sustainable Manufacturing | Supporting lower-energy robotic operations and Industry 5.0 objectives. |

*These are potential application areas; physical deployment and industrial performance have not yet been validated.*

---

# 📈 Experimental Figures & Visualization

To make the repository easier to review, include relevant figures from the manuscript where permitted.

Recommended visualizations include:

- 🤖 PyBullet simulation environment of the 6-DOF manipulator.
- 🏗️ Overall PG-SAC system architecture.
- 🧠 SAC agent–environment interaction workflow.
- 📉 Training reward and convergence curves.
- ⚡ Cumulative energy consumption comparison.
- 🔄 Mechanical jerk profiles.
- 🎯 Trajectory tracking error curves.
- ⚙️ Joint torque distribution.
- 📊 Optuna reward-weight optimization results.
- 🌐 Multi-target tracking and energy profiles.

Use descriptive filenames and reference them in the README when the images have been added to the repository.

---

# 📚 Publication Status

This repository accompanies the following research manuscript:

**Physics-Guided Energy-Efficient Trajectory Planning for a 6-DOF Industrial Manipulator Using Soft Actor-Critic Reinforcement Learning**

**Journal:** *Robotics and Autonomous Systems*  
**Status:** Under revision

The research introduces the PG-SAC framework and evaluates its performance in a physics-based simulation environment.

*The manuscript has not been described here as an accepted or published journal article.*

---

# 📝 Citation

If you refer to this research in academic work, use the following provisional BibTeX entry:

```bibtex
@article{barman2026pgsac,
  author  = {Barman, Shamol Chandra and Zonayed, MD and Siyer, MD},
  title   = {Physics-Guided Energy-Efficient Trajectory Planning for a 6-DOF Industrial Manipulator Using Soft Actor-Critic Reinforcement Learning},
  journal = {Robotics and Autonomous Systems},
  year    = {2026},
  note    = {Manuscript under revision}
}
```

Updated publication status and bibliographic details will be changed soon. DOI, volume, issue, or page numbers will be added when available.

---

# 👥 Authors

### Shamol Chandra Barman
Department of Mechatronics Engineering, World University of Bangladesh, Bangladesh.

### MD Zonayed
Department of Mechatronics Engineering, World University of Bangladesh, Bangladesh.

Research interests include robotics, artificial intelligence, reinforcement learning, energy-efficient control, and intelligent mechatronic systems.

- [Google Scholar](https://scholar.google.com/citations?user=oDymy0IAAAAJ&hl=en)
- [LinkedIn](https://www.linkedin.com/in/md-zonayed-8966a8276)
- [ResearchGate](https://www.researchgate.net/profile/Md-Zonayed)
- [GitHub](https://github.com/md-zonayed)

### MD Siyer
Department of Mechatronics Engineering, World University of Bangladesh, Bangladesh.

---

# ⭐ Explore & Cite This Research

Researchers interested in this work are encouraged to explore the available implementation, inspect the reported simulation results, and cite the manuscript when building on its methods.

Contributions that improve reproducibility, reward design, energy modeling, statistical validation, and sim-to-real transfer are particularly relevant to the future development of this framework.

---

<p align="center">
  <b>🤖 Toward sustainable industrial robotics through physics-guided reinforcement learning, energy-aware planning, and smoother manipulator control.</b>
</p>
