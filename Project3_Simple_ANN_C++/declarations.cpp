//including all the header files required
#include "main.h"                                                                     
#include <fstream>
#include <boost/algorithm/string.hpp>


//defining a function readCSV() to read the dataset
std::vector<std::vector<std::string>> Declarations::readCSV() {                   
    std::fstream file(this->dataset);                                                               //opening the file using fstream and giving object name as file 
    std::vector<std::vector<std::string>> dataString;                                               //declaring the variable dataString to store all the rows and colums    
    std::string line;                                                                               //declaring variable line

    if (!file.is_open()) {                                                                          //checking whether file is open or not
        std::cout<<"Could not open file: " <<this->dataset;                                         //if not , printing error statement
    }

    while (std::getline(file, line)) {                                                              //runing while loop till last line
        std::vector<std::string> vec;                                                               //declaring variable vec
        boost::algorithm::split(vec, line, boost::is_any_of(this->delimiter));                      //spliting the line with delimeter and storing it in a vec
        dataString.push_back(vec);                                                                  //pushing the vector to the variable dataString
    }

    file.close();                                                                                  //closing the file
    return dataString;                                                                             //returning the dataString
}

//defining the function createTensor  

torch::Tensor Declarations::createTensor() {                                                      
    std::vector<std::vector<std::string>> data = readCSV();                                      //calling the readCSV function to read dataset
    int rows = data.size();                                                                      //storing the size of rows to rows variable
    int cols = data[0].size();                                                                   //storing the size of 1 rows, i.e, columns in cols variable

    std::vector<double> flattenedData;                                                           //declaring a variable flattenedData
    flattenedData.reserve(rows * cols);                                                          //reversing the data 
    for (const auto& row : data) {                                                               //for loop to run till last row
        for (const auto& value : row) {                                                          //for loop to run till last value in a row
            flattenedData.push_back(std::stod(value));                                          //here we are converting string to double and storing in flatteneddata
        }
    }

    torch::Tensor tensor = torch::from_blob(flattenedData.data(), {rows, cols}, at::kDouble).clone();  //converting the data to tensors by giving the shape and type .
    return tensor;
}

//calling splitFeaturesAndTargets function

std::pair<torch::Tensor, torch::Tensor> Declarations::splitFeaturesAndTargets(const torch::Tensor& data) {
    int labelIndex = data.size(1) - 1;                                                                      //storing the size of features to the variable labelIndex           
    auto features = data.index({torch::indexing::Slice(), torch::indexing::Slice(0, labelIndex)});          //slicing all the values except the last column to features   
    // std::cout<<torch::indexing::Slice(0, labelIndex);
    // std::cout<<torch::indexing::Slice();  
    auto labels = data.index({torch::indexing::Slice(), labelIndex});                                      //slicing the values of Outcome column alone to labels
    return {features, labels};                                                                             //returning the features and labels
}


//calling funciton trainTestSplit
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Declarations::trainTestSplit(const torch::Tensor& features, const torch::Tensor& labels, double trainSize) {

    int totalSize = features.size(0);                                                                       //storing features size of 1 row to totalsize
    int trainSizeRows = (int)(totalSize * trainSize);                                                       //storing the trainsize of rows to trainsizerows
    int testSizeRows = totalSize - trainSizeRows;                                                           //storing the testsizerows to testsizerows variable

    torch::manual_seed(1);                                                                                  // Setting random seed for producing same set of data 

    auto indices = torch::randperm(totalSize);                                                              //randomly genertes the rows numbers and stores to indices variable
    // std::cout<<indices;
    auto train       = indices.slice(0, 0, trainSizeRows);                                                  //slicing the trainrows to trainIndices
    auto test        = indices.slice(0, trainSizeRows, totalSize);                                          //slicing the testrows to testIndices

    auto X = features.index_select(0, train);                                                               //storing features for training to X variable
    auto Y = (labels.index_select(0, train));                                                               //storing labels for training to Y variable
    auto x = features.index_select(0, test);                                                                //storing features for testing to x variable      
    auto y = labels.index_select(0, test);                                                                  //storing labels for testing to y variable

    return {X, Y, x, y};                                                                                    //returning all the splitted data
}



std::pair<torch::Tensor, torch::Tensor> Declarations::add_noise(const torch::Tensor& data, const torch::Tensor& labels) {
    float std = 0.2;
    int mean = 0;
    auto noisy_data = data.clone();
    auto noise = torch::randn(noisy_data.sizes()) * std + mean;
    noisy_data = noise + noisy_data;
    // return {noisy_data, labels};
    return {{torch::cat({data, noisy_data},0)}, {torch::cat({labels, labels},0)}};
}
