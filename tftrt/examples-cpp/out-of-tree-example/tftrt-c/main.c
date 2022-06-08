#include <stdio.h>
#include <tensorflow/c/c_api.h>

void DataDeallocator(void* data, size_t a, void* b)
{
    printf("Executing DataDeallocator\n");

    if (data)
        free(data);

    data = NULL;
}

int main(void)
{
    printf("Hello from TF C Library, version # %s\n", TF_Version());

    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();

    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Buffer* run_opts = NULL;

    const char* saved_model_dir = "../toy_model/converted";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    TF_Session* session = TF_LoadSessionFromSavedModel(session_opts,
                                    run_opts,
                                    saved_model_dir,
                                    &tags,
                                    ntags,
                                    graph,
                                    NULL,
                                    status);

    if (!session)
    {
        printf("ERROR: unable to load session\n");
        exit(1);
    }

    if (TF_GetCode(status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s", TF_Message(status));
    }


    // Get input tensors to the graph
    int num_inputs = 1;

    TF_Output* input = (TF_Output *) malloc(sizeof(TF_Output) * num_inputs);
    TF_Output in_tensor = { TF_GraphOperationByName(graph, "serving_default_conv2d_input"), 0 };

    if(in_tensor.oper == NULL)
    {
        printf("ERROR: Failed TF_GraphOperationByName serving_default_conv2d_input\n");
        exit(1);
    }
    else
    {
        printf("TF_GraphOperationByName serving_default_conv2d_input OK\n");
    }

    input[0] = in_tensor;

    // Get graph output tensors
    int num_outputs = 1;

    TF_Output* output = (TF_Output *) malloc(sizeof(TF_Output) * num_outputs);
    // Use the name of the output tensor as reported by saved_model_cli for the converted model.
    TF_Output out_tensor = { TF_GraphOperationByName(graph, "PartitionedCall"), 0 };

    if (out_tensor.oper == NULL)
    {
        printf("ERROR: Failed TF_GraphOperationByName PartitionedCall\n");
        exit(1);
    }
    else
    {
        printf("TF_GraphOperationByName PartitionedCall OK\n");
    }

    output[0] = out_tensor;

    // Allocate data for inputs & outputs
    TF_Tensor** input_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*) * num_inputs);
    TF_Tensor** output_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*) * num_outputs);

    int ndims = 4;
    int64_t dims[] = {1, 28, 28, 1};
    const size_t size = 28*28*1;
    float* data = (float *) malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
    {
        data[i] = 1.00;
    }

    size_t ndata = size * sizeof(float);
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &DataDeallocator, 0);

    if (input_tensor != NULL)
    {
        printf("TF_NewTensor OK\n");
    }
    else
    {
        printf("ERROR: Failed TF_NewTensor\n");

        if (data)
            free(data);

        exit(1);
    }

    input_values[0] = input_tensor;

    // Run the session
    TF_SessionRun(session,
                    NULL,
                    input, input_values, num_inputs,
                    output, output_values, num_outputs,
                    NULL, 0,
                    NULL,
                    status);

    if (TF_GetCode(status) == TF_OK)
    {
        printf("TF_SessionRun OK\n");
    }
    else
    {
        printf("TF_SessionRun ERROR: %s\n", TF_Message(status));
        exit(1);
    }

    void* buff = TF_TensorData(output_values[0]);
    float* offsets = buff;
    printf("results tensor : \n");
    printf("%f\n", offsets[0]);

    // Free all allocated resources
    TF_DeleteGraph(graph);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteStatus(status);

    return 0;
}
