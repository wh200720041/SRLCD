#pragma once
#ifndef LOOP_CLASS
#define LOOP_CLASS 

class Loop {
public:
	Loop(int current_frame_in, int loop_frame_in, float similarity_in);
	int current_frame;
	int loop_frame;
	float similarity;
};


#endif