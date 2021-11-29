# Save actual folder
FOLDER=${PWD}

# -----------------------------------------------------------------------------
# If you want a command to always run, put it here
# Carry out specific functions when asked to by the system 
case "$1" in
    start)
    echo  "Starting Robot"

        #  ----------------------------------------------------------------------
        # Source local enviroment variables
        source "/workspace/planner/configs/env_vars.sh"

        #  ----------------------------------------------------------------------
        # Delete previous workspaces
        if [ "$DELETE_BUILD" = "1" ] 
        then
            echo  [WARN]: "ROS2 Removing old shit ... "
            rm -r /workspace/planner/ros2/install || true
            rm -r /workspace/planner/ros2/build || true
            rm -r /workspace/planner/ros2/log || true
            sleep 2 && clear 
        fi

        #  ----------------------------------------------------------------------
        # Build ROS2 packages
        cd /workspace/planner/ros2/

        if [ ! "$2" == "no-build" ]; then
            . /opt/ros/galactic/setup.sh
            echo  "[INFO]: ROS2 Building new stuff ... "
            #colcon build --symlink-install --packages-skip interfaces
            colcon build --symlink-install
            echo  "[INFO]: ROS2 Build successful ... "
        fi
        echo  "[INFO]: ROS2 sourcing ... "
        source /workspace/planner/ros2/install/setup.sh

        #  ----------------------------------------------------------------------
        #  ROS2 Launching
        echo  "[INFO]: ROS2 launching ... "
        ros2 launch "/workspace/planner/configs/planner.launch.py"

esac

exit 0

#----------------------------------------------------------------------
# Return to scripts/ folder
cd ${PWD}