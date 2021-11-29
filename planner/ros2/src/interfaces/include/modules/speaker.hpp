/*! @package speaker
    Code Information:
        Maintainer: Eng. Davidson Daniel Rojas Cediel
        Mail: davidson@kiwibot.com
        Kiwi Campus / AI & Robotics Team
*/

#ifndef SPEAKER_H_INCLUDED
#define SPEAKER_H_INCLUDED

// ROS Node
#include <rclcpp/rclcpp.hpp>

// ROS2 Messages
// #include "std_msgs/msg/int8.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int8.hpp"

#include "utils/console.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <alsa/asoundlib.h>
#include <pthread.h>
#include <stdio.h>

using std::placeholders::_1;
class Speaker : public rclcpp::Node
{
    typedef void *(*THREADFUNCPTR)(void *);

public:
    /*!
        Speaker node constructor.
        @param options RCLCPP Node options.
        @return None
    */
    Speaker(rclcpp::NodeOptions &options);
    /*!
        Speaker node destructor.
        @see interfaces_node
    */
    ~Speaker();

    /*!
        CallBack trigger by the subscriber
        @param _msg std_msgs::msg::Int8::SharedPtr Message to with data to choose
        the track-sound to play
    */
    void speakerCb(const std_msgs::msg::Int8::SharedPtr _msg);

    /*!
    Thread function to play the sounds
    */
    void *PlaySound();

    /*!
    Thread function to play ambient sound
    */
    void *AmbientSound();

private:
    /*!
        Subscriber to trigger the Sound
    */
    /********************************************
    * USE THIS AMAZING SUBSCRIBER
    ********************************************/

    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr m_speaker_sub;

    /********************************************
    * END CODE 
   ********************************************/

    /*!
        Publisher to ensure the sound as Done
    */
    /********************************************
    * DEFINE THIS AMAZING PUBLISHER
    ********************************************/
    //[FEAT]:Create the publisher for the signal of the node is ready to play other sound
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr m_done_pub ;

    /********************************************
    * END CODE 
    ********************************************/

    /*!
        Attributes
    */
    std::string m_conf_path = getEnv("CONF_PATH", "/");
    std::string m_sound_device = (getEnv("SOUND_DEVICE", "sysdefault:CARD=PCH"));
    unsigned int channels = std::stoi(getEnv("SPEAKER_CHANNELS", "2"));
    unsigned int rate = std::stoi(getEnv("SPEAKER_RATE", "44100"));

    /*
        Path
    */
    std::string m_path;

    /*
    Handle the sound device
    */
    snd_pcm_t *pcm_handle;
    snd_pcm_hw_params_t *params;
    char *buff = nullptr;
    char *buff_ambient = nullptr;

    /* Prepare Device to play sounds */
    int pcm;
    snd_pcm_uframes_t frames;
    int buff_size;
    int buff_size_ambient;
    int readfd, readval, readfd_ambient, readval_ambient = 0;
    int status = 0;
    int ambient = 0;
    int m_pause = 0;
    int m_multi_sound = 1;
    /* Thread to play the sound */
    pthread_t pthread_id;
    pthread_t pthread_id_ambient;
};
#endif
