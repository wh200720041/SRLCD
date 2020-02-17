// Copyright (c) <2016>, <Nanyang Technological University> All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef TIMER_H
#define TIMER_H
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
using namespace std::chrono;

namespace Jeffsan{

class Timer
{
public:
	Timer()
	{
		tic();
	}

	double end()
	{
		time_span = duration_cast<duration<double>> (high_resolution_clock::now() - start);
		tic();
		return time_span.count();
	}

	void tic()
	{
		start = high_resolution_clock::now();
	}

	void toc(std::string section = " ")
	{
		end();
		printf("%s T: %g FPS: %gHz \r\n", section.c_str(), time_span.count(), 1/(time_span.count()));
	}

	void hz(std::string section = " ")
	{
		++num;
		time_span = duration_cast<duration<double>> (high_resolution_clock::now() - start);

		if (time_span.count() >= 1.0)
		{
			printf("%s: %f Hz\n", section.c_str(), double(num)/(time_span.count()));
			num = 0;
			tic();
		}
	}

private:
	high_resolution_clock::time_point start;
	duration<double> time_span;
	unsigned int num;
};

}
#endif 
