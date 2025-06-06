#include <ros/ros.h>
#include <std_msgs/String.h>

#include <sstream>
#include <cstdlib>

#include "tm_msgs/SetEvent.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "set_event_test");
    ros::NodeHandle event_demo;
    ros::ServiceClient client = event_demo.serviceClient<tm_msgs::SetEvent>("tm_driver/set_event");
    tm_msgs::SetEvent srv;

    srv.request.func = tm_msgs::SetEvent::Request::STOP;
    srv.request.arg0 = 0;
    srv.request.arg1 = 0;

    if (client.call(srv))                             
    {
        if (srv.response.ok) ROS_INFO_STREAM("SetEvent to robot");
        else ROS_WARN_STREAM("SetEvent to robot , but response not yet ok ");
    }
    else
    {
        ROS_ERROR_STREAM("Error SetEvent to robot");
        return 1;
    }

  //ROS_INFO_STREAM_NAMED("SetEvent", "shutdown.");  	
  return 0;
}