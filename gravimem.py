import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Literal

class GraviConstants:
    TRUTH_SCALE = 20.0
    INITIAL_CERTAINTY = 0.5
    MIN_CERTAINTY = 0.1
    MAX_CERTAINTY = 10.0
    MASTERY_THRESHOLD = 6.0
    
    GAIN_EASY = 1.5
    GAIN_GOOD = 0.8
    GAIN_HARD = 0.3
    LOSS_AGAIN = 2.0

    ALPHA = 0.25
    BETA = 0.40
    GAMMA = 0.10

    NEIGHBOR_RADIUS = 4.5
    CONFUSOR_RADIUS = 3.0

    DECAY_HALFLIFE_DAYS = 14
    THERMAL_NOISE_RATE = 0.05
    MAX_THERMAL_NOISE = 1.0
    
    MS_PER_DAY = 24 * 60 * 60 * 1000

@dataclass
class GraviConcept:
    id: str
    mind_embedding: List[float]
    truth_embedding: List[float]
    certainty: float = GraviConstants.INITIAL_CERTAINTY
    last_reviewed_at: float = 0

@dataclass
class GraviUpdateResult:
    new_mind_embedding: List[float]
    new_certainty: float

@dataclass
class RippleUpdate:
    id: str
    new_mind_embedding: List[float]



@dataclass
class SympatheticMeltUpdate:
    id: str
    certainty_loss: float

def euclidean_distance(a: List[float],b: List[float]) -> float: 
    if len(a) != len(b):
        return float('inf')
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def magnitude(vec: List[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in vec))

def add_vectors(a: List[float], b: List[float]) -> List[float]:
    return [xeuclidean_distance + y for x, y in zip(a, b)]

def subtract_vectors(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def scale_vector(vec: List[float], scalar: float) -> List[float]:
    return [x * scalar for x in vec]

def initialize_mind_embedding(dimensions: int) -> List[float]:
    return [(random.random() - 0.5) * 0.1 for _ in range(dimensions)]

def scale_truth_embedding(normalized_embedding: List[float]) -> List[float]:
    return scale_vector(normalized_embedding, GraviConstants.TRUTH_SCALE)

def calculate_selection_stress(concept: GraviConcept, all_concepts: List[GraviConcept]) -> float:
    error = euclidean_distance(concept.mind_embedding, concept.truth_embedding)
    stress_sum = 0
    count = 0
    
    sample_size = min(len(all_concepts), 50)
    step = len(all_concepts) // sample_size if len(all_concepts) > sample_size else 1
    
    for i in range(0, len(all_concepts), step):
        other = all_concepts[i]
        if other.id == concept.id:
            continue
            
        truth_dist = euclidean_distance(concept.truth_embedding, other.truth_embedding)
        
        if truth_dist < GraviConstants.NEIGHBOR_RADIUS:
            mind_dist = euclidean_distance(concept.mind_embedding, other.mind_embedding)
            stress_sum += abs(mind_dist - truth_dist)
            count += 1
            
    avg_topo_stress = stress_sum / count if count > 0 else 0
    plasticity = 1.0 / (max(concept.certainty, 0.1) + 0.1)
    
    return (error + avg_topo_stress) * plasticity

def process_correct_answer(
    target: GraviConcept, 
    all_concepts: List[GraviConcept], 
    grade: Literal['easy', 'good', 'hard'] = 'good'
) -> Tuple[GraviUpdateResult, List[RippleUpdate]]:
    c = GraviConstants
    
    gain = {
        'easy': c.GAIN_EASY,
        'good': c.GAIN_GOOD,
        'hard': c.GAIN_HARD
    }.get(grade, c.GAIN_GOOD)
    
    new_certainty = min(c.MAX_CERTAINTY, target.certainty + gain)

    effective_lr = c.ALPHA / math.sqrt(max(target.certainty, 0.5))
    direction = subtract_vectors(target.truth_embedding, target.mind_embedding)
    new_mind_embedding = add_vectors(target.mind_embedding, scale_vector(direction, effective_lr))

    ripple_updates = []
    
    for neighbor in all_concepts:
        if neighbor.id == target.id:
            continue

        truth_dist = euclidean_distance(target.truth_embedding, neighbor.truth_embedding)
        
        if truth_dist > c.NEIGHBOR_RADIUS:
            continue

        ripple_strength = math.exp(-truth_dist / c.NEIGHBOR_RADIUS)
        neighbor_lr = (c.GAMMA * ripple_strength) / math.sqrt(max(neighbor.certainty, 0.5))
        neighbor_dir = subtract_vectors(neighbor.truth_embedding, neighbor.mind_embedding)
        new_neighbor_mind = add_vectors(neighbor.mind_embedding, scale_vector(neighbor_dir, neighbor_lr))
        
        ripple_updates.append(RippleUpdate(neighbor.id, new_neighbor_mind))

    return GraviUpdateResult(new_mind_embedding, new_certainty), ripple_updates


def process_incorrect_answer(
    target: GraviConcept, 
    all_concepts: List[GraviConcept]
) -> Tuple[GraviUpdateResult, List[SympatheticMeltUpdate]]:
    c = GraviConstants

    new_certainty = max(c.MIN_CERTAINTY, target.certainty - c.LOSS_AGAIN)

    confusor = None
    min_dist = float('inf')
    target_mag = magnitude(target.mind_embedding)

    if target_mag > 1.0:
        for other in all_concepts:
            if other.id == target.id:
                continue
            
            if magnitude(other.mind_embedding) < 1.0:
                continue

            mind_dist = euclidean_distance(target.mind_embedding, other.mind_embedding)
            if mind_dist < c.CONFUSOR_RADIUS and mind_dist < min_dist:
                min_dist = mind_dist
                confusor = other

    new_mind_embedding = target.mind_embedding

    if confusor:
        repulsion_lr = c.BETA / math.sqrt(new_certainty)
        repel_dir = subtract_vectors(target.mind_embedding, confusor.mind_embedding)
        
        dist = max(min_dist, 0.1)
        norm_repel = scale_vector(repel_dir, 1.0 / dist)
        
        new_mind_embedding = add_vectors(new_mind_embedding, scale_vector(norm_repel, repulsion_lr))

    regression_strength = c.ALPHA / math.sqrt(new_certainty)
    new_mind_embedding = scale_vector(new_mind_embedding, 1.0 - (regression_strength * 0.5))

    melt_updates = []
    
    for neighbor in all_concepts:
        if neighbor.id == target.id:
            continue

        truth_dist = euclidean_distance(target.truth_embedding, neighbor.truth_embedding)
        if truth_dist > c.NEIGHBOR_RADIUS:
            continue

        melt_strength = math.exp(-truth_dist / c.NEIGHBOR_RADIUS) * 0.2
        loss = neighbor.certainty * melt_strength
        
        if loss > 0.1:
            melt_updates.append(SympatheticMeltUpdate(neighbor.id, loss))

    return GraviUpdateResult(new_mind_embedding, new_certainty), melt_updates


def apply_thermodynamic_decay(
    concepts: List[GraviConcept], 
    now_ms: float = None
) -> List[Tuple[str, float, List[float]]]:
    c = GraviConstants
    if now_ms is None:
        now_ms = time.time() * 1000

    updates = []
    half_life_ms = c.DECAY_HALFLIFE_DAYS * c.MS_PER_DAY

    for concept in concepts:
        if not concept.last_reviewed_at:
            continue

        elapsed_ms = now_ms - concept.last_reviewed_at
        if elapsed_ms < c.MS_PER_DAY:
            continue

        effective_elapsed = min(elapsed_ms, 60 * c.MS_PER_DAY)
        days = effective_elapsed / c.MS_PER_DAY

        decay_factor = math.exp(-0.693 * effective_elapsed / half_life_ms)
        new_certainty = max(c.MIN_CERTAINTY, concept.certainty * decay_factor)

        noise_mag = min(c.MAX_THERMAL_NOISE, c.THERMAL_NOISE_RATE * days)
        noise = [(random.random() - 0.5) * noise_mag for _ in concept.mind_embedding]
        
        new_mind_embedding = add_vectors(concept.mind_embedding, noise)
        
        updates.append((concept.id, new_certainty, new_mind_embedding))

    return updates
