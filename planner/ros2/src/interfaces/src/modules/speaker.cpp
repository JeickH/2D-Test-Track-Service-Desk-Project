/*! @package speaker
    Code Information:
        Maintainer: Eng. Davidson Daniel Rojas Cediel
        Mail: davidson@kiwibot.com
        Kiwi Campus / AI & Robotics Team
*/

#include "modules/speaker.hpp"

Speaker::Speaker(rclcpp::NodeOptions &options) : Node("speaker", "interfaces", options)
{
    RCLCPP_DEBUG(this->get_logger(), "Speaker Contructor");

    auto default_qos = rclcpp::QoS(rclcpp::SystemDefaultsQoS());
    //QOS profile compatibility on subscriber m_speaker_sub 
    auto sensor_data_qos = rclcpp::QoS(rclcpp::SensorDataQoS());

    // /* Publisher */

    /********************************************
    * USE THIS AMAZING PUBLISHER
    ********************************************/
    m_done_pub = this->create_publisher<std_msgs::msg::Bool>("/device/speaker/done", default_qos);
    /********************************************
    * END CODE 
    ********************************************/

    // Subscribers

    /********************************************
    * DEFINE YOUR AMAZING SUBSCRIBER 
    * Find Documentation here:
    * https://docs.ros.org/en/foxy/Tutorials/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html#write-the-subscriber-node
    ********************************************/
    //[FEAT]:Create the subscriber to reproduce a .wav file when receive a message
    m_speaker_sub= this->create_subscription<std_msgs::msg::Int8>("/device/speaker/command", sensor_data_qos, std::bind(&Speaker::speakerCb, this, _1)); 

    /********************************************
    * END CODE 

    /* Open the PCM device in playback mode */
    if (pcm = (snd_pcm_open(&pcm_handle, m_sound_device.c_str(), SND_PCM_STREAM_PLAYBACK, 0) < 0))
        RCLCPP_ERROR(this->get_logger(), "ERROR: Can't open \"%s\" PCM device. %s", m_sound_device, snd_strerror(pcm));

    /* Allocate parameters object and fill it with default values*/
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(pcm_handle, params);

    /* Set parameters */
    if (pcm = (snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED) < 0))
        RCLCPP_ERROR(this->get_logger(), "ERROR: Can't set interleaved mode. %s", snd_strerror(pcm));

    if (pcm = (snd_pcm_hw_params_set_format(pcm_handle, params, SND_PCM_FORMAT_S16_LE) < 0))

        RCLCPP_ERROR(this->get_logger(), "ERROR: Can't set format. %s\n", snd_strerror(pcm));

    if (pcm = (snd_pcm_hw_params_set_channels(pcm_handle, params, channels) < 0))
        RCLCPP_ERROR(this->get_logger(), "ERROR: Can't set channels number. %s\n", snd_strerror(pcm));

    if (pcm = (snd_pcm_hw_params_set_rate_near(pcm_handle, params, &rate, 0) < 0))
        RCLCPP_ERROR(this->get_logger(), "ERROR: Can't set rate. %s\n", snd_strerror(pcm));

    /* Write parameters */
    if (pcm = (snd_pcm_hw_params(pcm_handle, params) < 0))
        RCLCPP_ERROR(this->get_logger(), "ERROR: Can't set harware parameters. %s\n", snd_strerror(pcm));

    /* Resume information */
    RCLCPP_INFO(this->get_logger(), "PCM name: '%s'\n", snd_pcm_name(pcm_handle));

    RCLCPP_INFO(this->get_logger(), "PCM state: %s\n", snd_pcm_state_name(snd_pcm_state(pcm_handle)));

    snd_pcm_hw_params_get_period_size(params, &frames, 0);

    buff_size = frames * channels * 2;
    buff = (char *)malloc(buff_size);
    memset(buff, 0, buff_size);

    buff_size_ambient = frames * channels * 2;
    buff_ambient = (char *)malloc(buff_size_ambient);
    memset(buff_ambient, 0, buff_size_ambient);

    m_path = "/workspace/planner/media/audio/track";

    snd_pcm_start(pcm_handle);
    readfd_ambient = open((m_path + "0.wav").c_str(), O_RDONLY);
    ambient = pthread_create(&pthread_id_ambient, NULL, (THREADFUNCPTR)&Speaker::AmbientSound, this);
}

void Speaker::speakerCb(const std_msgs::msg::Int8::SharedPtr msg)
{
    /*
+       Note: Every sound-track should be named "track#"
        To Stop send a 0 message.
    */
    m_multi_sound = 0;
    snd_pcm_start(pcm_handle);
    if (msg->data > 0)
    {
        std::ifstream ifile;
        ifile.open(m_path + std::to_string(msg->data) + ".wav");
        if (ifile)
        {
            readfd = open((m_path + std::to_string(msg->data) + ".wav").c_str(), O_RDONLY);
            status = pthread_create(&pthread_id, NULL, (THREADFUNCPTR)&Speaker::PlaySound, this);
        }
        /********************************************
        * PLAY A DEFAULT SOUND IF NOT FOUND THE TRACK FILE
        ********************************************/
        //[FEAT]:define a default sound in case the first one is not founded
        else{
            readfd = open((m_path + "2.wav").c_str(), O_RDONLY);
            ambient = pthread_create(&pthread_id, NULL, (THREADFUNCPTR)&Speaker::PlaySound, this);
        }
        /********************************************
        * END CODE 
        ********************************************/

    }
    else
    {
        RCLCPP_DEBUG(this->get_logger(), "Sound stopped");
        m_pause = 1;
        m_multi_sound = 1;
    }
}

void *Speaker::PlaySound()
{
    if (readfd < 0)
    {
        RCLCPP_ERROR(this->get_logger(), "ERROR: .wav It's Empty");
        pthread_join(pthread_id, NULL);
    }
    /********************************************
    * PUBLISH YOUR AMAZING BOOL DATA
    * Take Care: in order to publish a Unique Pointer you need to pass std::move(msg) 
    * to the publish function. So you can't just pass the message inside the function, search for it :)
    * Documentation here:
    * https://docs.ros.org/en/foxy/Tutorials/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html#write-the-publisher-node
    ********************************************/
    //[FEAT]: publish a bool value to notify when is done the process of play a sound
    std_msgs::msg::Bool::UniquePtr msg(new std_msgs::msg::Bool());
    msg.reset(new std_msgs::msg::Bool());
    msg->data=false;
    m_done_pub->publish(std::move(msg));
    /********************************************
    * END CODE 
    ********************************************/
    while (readval = (read(readfd, buff, buff_size) > 0))
    {
        if (pcm = snd_pcm_writei(pcm_handle, buff, frames) == -EPIPE)
        {
            snd_pcm_prepare(pcm_handle);
        }
        else if (pcm < 0)
        {
            RCLCPP_ERROR(this->get_logger(), "ERROR: Writing to PCM device: %s", snd_strerror(pcm));
        }
    }
    m_multi_sound = 1;

    /********************************************
    * PUBLISH YOUR AMAZING BOOL DATA
    * Take Care: in order to publish a Unique Pointer you need to pass std::move(msg) 
    * to the publish function. So you can't just pass the message inside the function, search for it :)
    * Documentation here:
    * https://docs.ros.org/en/foxy/Tutorials/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html#write-the-publisher-node
    ********************************************/
    // This is just for clean the variable name and re-initialize it.
    //[FEAT]: publish a bool value to notify when is done the process of play a sound
    msg.reset(new std_msgs::msg::Bool());
    msg->data=true;
    m_done_pub->publish(std::move(msg));
    /********************************************
    * END CODE 
    ********************************************/
}

void *Speaker::AmbientSound()
{
    while (true)
    {
        if (!m_pause)
        {
            readfd_ambient = open((m_path + "0.wav").c_str(), O_RDONLY);
            if (readfd_ambient < 0)
            {
                RCLCPP_ERROR(this->get_logger(), "ERROR: .wav It's Empty");
                pthread_join(pthread_id_ambient, NULL);
            }
            while (readval_ambient = (read(readfd_ambient, buff_ambient, buff_size_ambient) > 0))
            {
                if (m_multi_sound)
                {
                    if (pcm = snd_pcm_writei(pcm_handle, buff_ambient, frames) == -EPIPE)
                    {
                        snd_pcm_prepare(pcm_handle);
                    }
                    else if (pcm < 0)
                    {
                        RCLCPP_ERROR(this->get_logger(), "ERROR: Writing to PCM device: %s", snd_strerror(pcm));
                    }
                }
            }
        }
    }
}

/* Class Destructor to clean up everything*/
Speaker::~Speaker()
{
    pthread_kill(pthread_id, 15);
    free(buff);
    pthread_kill(pthread_id_ambient, 15);
    free(buff_ambient);
    snd_pcm_close(pcm_handle);
}