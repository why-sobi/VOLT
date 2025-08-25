#pragma once

class Optimizer {
protected:
	float learning_rate;																		// How big the jumps in weight update should be
	float weight_decay;																			// To penalize large weights (L2 normalization)
	float momentum;																				// beta-1 for Adam
	float epsilon;																				// small value so no dividing my zero
};