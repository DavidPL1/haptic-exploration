cmake_minimum_required(VERSION 3.0.2)
project(haptic_exploration)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  mujoco_ros
  mujoco_contact_surface_sensors
  mujoco_ros_mocap
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
    CATKIN_DEPENDS   
      rospy 
      std_msgs 
      std_srvs
      mujoco_ros
      mujoco_contact_surface_sensors
      mujoco_ros_mocap
)

#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  # scripts/sample_simple.py
  # scripts/sample_composite.py
  scripts/ycb_recorder
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY launch config assets
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)