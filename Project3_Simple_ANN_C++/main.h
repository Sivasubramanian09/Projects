//Including all the Header files required

#include <torch/torch.h>                                           
#include <fstream>                                                 
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <stdexcept>
#include <iostream>
#include <random>


//Creating a Class named Declarations
class Declarations {
public:
    Declarations(const std::string& dataset, const std::string& delimiter)          //calling constructor
        : dataset(dataset), delimiter(delimiter) {}

    //Declaring the Required Functions
    std::vector<std::vector<std::string>> readCSV();                 
    torch::Tensor createTensor();
    std::pair<torch::Tensor, torch::Tensor> splitFeaturesAndTargets(const torch::Tensor& data);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> trainTestSplit(const torch::Tensor& features, const torch::Tensor& labels, double trainSize);
    std::pair<torch::Tensor, torch::Tensor> add_noise(const torch::Tensor& data, const torch::Tensor& labels);

    //decalring the variables dataset and delimeter
private:
    std::string dataset;
    std::string delimiter;
};


//Creating a class called ANN and inheriting the torch::nn:Module
class ANN : public torch::nn::Module {
public:
    // calling Constructor
    ANN(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        // Defining the layers of the Neural Network
        input_layer = register_module("input_layer", torch::nn::Linear(input_size, hidden_size));
        hidden_layer = register_module("hidden_layer", torch::nn::Linear(hidden_size, hidden_size));
        output_layer = register_module("output_layer", torch::nn::Linear(hidden_size, output_size));

        // Set the data type of all layers to match the data type of the input tensor
        input_layer->to(torch::kDouble);
        hidden_layer->to(torch::kDouble);
        output_layer->to(torch::kDouble);
    }

    // Defining the forward pass
    torch::Tensor forward(torch::Tensor x) {
        // Apply layers with activation functions
        x = torch::relu(input_layer->forward(x));
        x = torch::relu(hidden_layer->forward(x));
        x = torch::sigmoid(output_layer->forward(x));
        return x;
    }

    //Initializing each layers to the null pointer 
private:
    torch::nn::Linear input_layer{nullptr}, hidden_layer{nullptr}, output_layer{nullptr};
};
