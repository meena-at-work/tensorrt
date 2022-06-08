#pragma once

#include <string>

class NNRunner
{
    public:
        NNRunner(const std::string& model_path);
        ~NNRunner();
        // void LoadSavedModel();
        // void Calibrate();
        // void SetupInputTensors();
        // void SetupOutputTensors();
        // void RunInference();

    private:
        // TF_Graph* graph = TF_NewGraph();
        // TF_Status* status = TF_NewStatus();

        // TF_SessionOptions* session_opts = TF_NewSessionOptions();
        // TF_Buffer* run_opts = NULL;

};