/ Copyright (c) <2017>, <Nanyang Technological University> All rights reserved.

// This file is part of correlation_flow.

//     correlation_flow is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.

//     Foobar is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.

//     You should have received a copy of the GNU General Public License
//     along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

#include <ros/ros.h>
#include "correlation_flow/correlation_flow.h"

using namespace std;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "correlation_flow_node");

    ros::NodeHandle nh("~");

    CorrelationFlow cf(nh);

    image_transport::ImageTransport it(nh);
  	
  	image_transport::Subscriber sub = it.subscribe("camera/image", 1, &CorrelationFlow::callback, &cf);

    ros::spin();

    return 0;
}
