# pragma once

#include <iostream>
#include <vector>

template <typename T, typename U>
class Pair {
public:
	T first;  // First element of the pair
	U second; // Second element of the pair
	
	// Constructor to initialize the pair
	Pair(const T& first, const U& second) : first(first), second(second) {}
	
	// Default constructor
	Pair() : first(T()), second(U()) {}
	
	// Destructor
	~Pair() {
		// No dynamic memory allocation, so nothing special to do here
	}
};	