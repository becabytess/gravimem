# GRAVIMEM: Gravitational Memory Physics
> **Where Truth Has Gravity.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GRAVIMEM** is a breakthrough algorithm that treats knowledge not as a "scheduled task" (like Spaced Repetition), but as a **physical object in high-dimensional space**.

It replaces the 30-year-old "timer" paradigm (Anki, SM-2, FSRS) with a **Geometric Physics Simulation** of your mind.

---

## 🌌 The Paradigm Shift

### The Old Way: Time-Based (Like Anki)
*   *"You answered correctly. See you in 3 days."*
*   **Problem**: It ignores *context*. It treats "Quantum Mechanics" the same as "French Vocabulary".

### The GRAVIMEM Way: Physics + SGD
Gravimem is effectively running **Stochastic Gradient Descent (SGD)** on the human brain, modeled via physics.

*   **Loss Function**: `Distance(Mind, Truth)`. We minimize the error between your mental model and reality.
*   **Mass (Certainty)**: Acts as an inverse **Learning Rate**.
    *   **New Concepts (Low Mass)** have a high learning rate (Plasticity).
    *   **Mastered Concepts (High Mass)** have a low learning rate (Stability).
*   **Forces (Gradients)**:
    *   **Attraction**: Correct answers are gradient steps toward the Truth.
    *   **Repulsion**: Confusion creates specific gradients pushing concepts apart.
    *   **Backpropagation**: Corrections ripple through the graph, adjusting related concepts automatically.
    *   **Entropy**: Time doesn't just "reset a timer"; it adds *Thermal Noise* (jitter) and evaporates *Mass* (certainty).

---

## 🧠 Neuroscience Basis: Interference vs. Decay

Cognitive science posits two main causes of forgetting:

1.  **Decay Theory**: Memory fades simply due to the passage of time. (Modeled by Anki/SM-2).
2.  **Interference Theory**: Memory is corrupted by competing, similar information. (Ignored by traditional apps).

**GRAVIMEM is the first algorithm to mathematically model Interference Theory.**

*   **Handling Decay**: We use **Thermodynamic Entropy**. Over time, your knowledge concept loses Mass (Certainty) and jitters (Thermal Noise), naturally drifting away from truth.
*   **Handling Interference**: We use **Electrostatic Repulsion**. If you confuse "Stalactite" with "Stalagmite", Gravimem doesn't just schedule a review; it applies a repulsive force vector that physically pushes the **vector embeddings** of these two concepts apart in your latent space, reducing future interference.

---

## 🧠 Core Concepts

### 1. Elastic Manifold (Latent Space Optimization)
Gravimem projects your knowledge onto a high-dimensional manifold of **vector embeddings**. When you learn, you aren't just flipping a bit; you are **optimizing the vector positions** in this latent space. 

If you learn that *Lion* is a *Cat*, the algorithm applies a gradient update that pulls the entire *Feline* cluster **embeddings** into better alignment. **You learn faster because learning one thing physically moves related vector embeddings closer to mastery.**

### 2. Mass & Inertia (Adaptive Learning Rate)
In traditional algorithms, a "Review" is just a counter. In Gravimem, it is **Mass Accretion**.
*   **New Concepts** are gases (Low Mass / High LR). They move fast.
*   **Mastered Concepts** are solids (High Mass / Low LR). They are stable.

### 3. Sympathetic Melt (Unfreezing Layers)
If you suddenly fail a core concept you thought you knew, Gravimem triggers a **Sympathetic Melt**. The shockwave propagates through the graph, lowering the certainty of dependent concepts.
*   *ML Analogy*: This is like "Unfreezing" layers in a neural network to allow them to be relearned after a foundational error is detected.

---

## ⚡ Quick Start

GRAVIMEM is a pure mathematical kernel. It has **zero dependencies**.

```python
from gravimem import GraviConcept, process_correct_answer, GraviConstants

# 1. Define a Concept (e.g., from OpenAI/Cohere embeddings)
concept = GraviConcept(
    id="node_123",
    truth_embedding=[0.1, 0.5, ...], # Where it should be
    mind_embedding=[0.0, 0.0, ...]   # Where you are (starts near origin)
)

# 2. User answers correctly (Grade: "Good")
# The concept attracts toward truth, gains mass, and pulls neighbors!
result, ripples = process_correct_answer(concept, all_concepts, grade="good")

print(f"New Certainty: {result.new_certainty}") # Increased Mass
print(f"Ripples: {len(ripples)} related concepts improved!")
```

## 📉 Comparison

| Feature | Spaced Repetition (SM-2/FSRS) | GRAVIMEM (Physics) |
| :--- | :--- | :--- |
| **Model** | Timer / Scheduler | Physics / Vector Space |
| **Context** | None (Independent Cards) | **Semantic Graph** |
| **Interconnectedness** | None | **Hebbian Ripples** (Chain Reactions) |
| **Failure Mode** | Reset Interval to 0 | **Electrostatic Repulsion** (Push away from confusion) |
| **Decay** | Fixed Curve | **Thermodynamic Entropy** (Jitter + Evaporation) |

## 📜 License

MIT. Core engine written , more coming soon

# gravimem
