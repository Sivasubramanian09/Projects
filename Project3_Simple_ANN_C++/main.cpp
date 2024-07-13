//Including all header files

/*Task -2 
Use this example - https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/ to understand pytorch NNs.
Augment the dataset by perturbing the data fields with gaussian noise. Train/test/validate. Take a screencast walkthrough of the full code,
including your understanding of how the NN is implemented.
*/

#include "main.h"
#include <iostream>
#include <vector>

//main function
int main() {
    std::string file = "/home/siva/cpp/Programs/Task2/Task2_new/pima-indians-diabetes.data.csv";            //storing the file path to variabe filePath
    std::string delimiter = ",";                                                                            //storing the delimeter to the variable delimeter

    Declarations dec(file, delimiter);                                                                      //defining a object for the class as dec
    torch::Tensor tensor = dec.createTensor();                                                              //calling the function createTensor and assigning to tensor variable

    auto [features, targets] = dec.splitFeaturesAndTargets(tensor);                                         //calling function splitFeaturesAndTargets ang passing the tensor value and assigning it to features and labels
    
    std::cout<<"\nSize of Features :" <<features.sizes()<<std::endl;
    std::cout<<"\nSize of Targets :" <<targets.sizes()<<std::endl;

    // std::pair<torch::Tensor, torch::Tensor> result= dec.splitFeaturesAndTargets(tensor);                
    // torch::Tensor features = result.first;
    // torch::Tensor labels = result.second;   

    auto [trainFeatures, trainLabels, X_test, Y_test] = dec.trainTestSplit(features, targets, 0.8);         //spliting 80% of the data into train and remaining to test data using traintestsplit function
    auto [X_train, Y_train, ValFeatures, ValLabels] = dec.trainTestSplit(trainFeatures, trainLabels, 0.8);  //spliting 80% of the data into training and remaining to validation data using traintestsplit function

   //printing the shapes of all the splited data

    std::cout <<"\n================Sizes of all the Splitted Data: =============== \n";
    std::cout << "Training features:\n" << X_train.sizes() << std::endl;
    std::cout << "Training labels:\n" << Y_train.sizes() << std::endl;
    std::cout << "Testing features:\n" << X_test.sizes() << std::endl;
    std::cout << "Testing labels:\n" << Y_test.sizes() << std::endl;
    std::cout << "Validation features:\n" <<ValFeatures.sizes() <<std::endl;
    std::cout << "Validation Labels:\n" <<ValLabels.sizes() <<std::endl;

    int input_size = 8;                                                                                     //assigning input_size as 8                                       
    int hidden_size = 12;                                                                                   //assigning hidden_size as 12
    int output_size = 1;                                                                                    //assigning output_size as 1

    ANN model(input_size, hidden_size, output_size);                                                //creating an object called model for the class ANN 


    // Loss function and optimizer
    torch::nn::BCELoss loss_fn;                                                                    //setting loss fucntion as BCELoss
    auto optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(0.01));      //setting optimizer as Adam

    // Training the model 
    int n_epochs = 200;                                                                           //assigning n_epochs as 200
    int batch_size = 10;                                                                          //assigning batch_size as 10

    std::cout<<"\n\n========= TRAINING THE MODEL========= \n\n";

    for (int epoch = 0; epoch < n_epochs; epoch++) {                                              //for loop to run till n_epochs 
        model.train();                                                                            //training the model
        float train_loss = 0.0;                                                                   //declaring a variable train_loss as 0.0 
        for (size_t i = 0; i < X_train.size(0); i += batch_size) {                                //for loop to run till size of the X_train
            auto X_batch = X_train.narrow(0, i, std::min((size_t)(batch_size), X_train.size(0) - i));    //storing the batch of X_train elements in X_batch
            auto y_batch = Y_train.narrow(0, i, std::min((size_t)(batch_size), Y_train.size(0) - i));    //storing the batch of Y_train elements in Y_batch

            y_batch = y_batch.view({-1, output_size});                                            //reshaping the Y_batch and storing in y_batch 

            // optimizer.zero_grad();                                                                //clearing the gradients
            auto y_pred = model.forward(X_batch);                                                 //sending the Xbatch to the model to do forward pass
            auto loss = loss_fn(y_pred, y_batch);                                                 //finding the loss
            optimizer.zero_grad();  
            loss.backward();                                                                      //doing back propagation
            optimizer.step();                                                                     //updating the weights and biases
            train_loss += loss.item<float>();                                                     //adding the each batch loss to the train_loss variable
        }

        train_loss /= (X_train.size(0) / batch_size);                                             //finding the train_loss for 1 epoch


        float val_loss = 0.0;                                                                     //declaring a variable val_loss as 0.0
        for (size_t j = 0; j < ValFeatures.size(0); j += batch_size) {                            //for loop to run till size of the valFeatures size
            auto X_batch = ValFeatures.narrow(0, j, std::min(static_cast<size_t>(batch_size), ValFeatures.size(0) - j));  //storing the batch of ValFeaturs elements in X_batch
            auto y_batch = ValLabels.narrow(0, j, std::min(static_cast<size_t>(batch_size), ValLabels.size(0) - j));      //storing the batch of ValLabels elements in X_batch

            y_batch = y_batch.view({-1, output_size});                                                   //reshaping the Y_batch and storing in y_batch

            auto y_pred = model.forward(X_batch);                                                        //sending the Xbatch to the model to do forward pass              
            auto loss = loss_fn(y_pred, y_batch);                                                        //finding the loss 
            val_loss += loss.item<float>();                                                              //adding the each batch loss to the val_loss variable
        }

        val_loss /= (ValFeatures.size(0) / batch_size);                                                  //finding the val_loss for 1 epoch
        
        //printing the train_loss and val_loss for every 10 epoch
        if (epoch % 10 == 0){
            std::cout << "Epoch: " << epoch << "  Train Loss: " << train_loss <<" Val loss: "<<val_loss<< std::endl;
        }
    }


    model.eval();                                                                                        //evaluating the model
    auto y_pred = model.forward(X_test);                                                                 //checking the model by sending the test data
    Y_test = Y_test.view({-1, output_size});                                                             //reshaping the Y_test
    auto test_loss = loss_fn(y_pred, Y_test).item<float>();                                              //finding the test loss
    std::cout << "\nTest loss: " << test_loss << std::endl;                                                //printing the test loss

    auto accuracy = ((y_pred.round() == Y_test).to(torch::kFloat32).mean().item<float>());               //finding the accuracy of the model 
    std::cout << "\nAccuracy: " << accuracy << std::endl;                                                  //printing the accuracy of the model

    std::cout<<"\nPREDICTIONS:\n";

    auto predictions = (y_pred > 0.5).to(torch::kInt32);                                                 //printing the predictions                                         
    for (int i = 0; i < 5; i++) {                                                                        //running for loop to run till 5 elements
        std::cout<< y_pred[i].item<float>() << " => " << predictions[i].item<int>() << " (expected " << Y_test[i].item<float>() << ")" << std::endl;  //printing the outcome
    }

    std::cout<<"\n\n========SHAPE OF DATA AFTER AUGMENTING==========\n";
    // Adding Gausian Noise
    auto [aug_X_data, aug_Y_data] = dec.add_noise(X_train, Y_train);                                     //calling add_noise function and passing X_train and Y_train
    std::cout << "\nAugmented X data shape: " << aug_X_data.sizes() << std::endl;                            //printing the sizes of X_data after adding gausian noise
    std::cout << "\nAugmented Y data  shape: " << aug_Y_data.sizes() << std::endl;                         //printing the sizes os Y_data after adding gausian noise

    // std::cout<<aug_X_data<<std::endl;
    // std::cout<<aug_Y_data<<std::endl;
    ANN aug_model(8,12,1);                                                                              //Creating an object after augmentation

    // Loss function and optimizer
    torch::nn::BCELoss aug_loss_fn;                                                                    //setting loss fucntion as BCELoss 
    auto aug_optimizer = torch::optim::Adam(aug_model.parameters(), torch::optim::AdamOptions(0.01));  //setting optimizer as Adam

    // Training the model
    int epochs = 100;                                                                                  //assigning epochs as 100
    int aug_batch_size = 10;                                                                           //assigning aug_batch_size as 10

    std::cout<<"\n\n========= TRAINING THE MODEL========= \n\n";
    std::cout <<"===============AFTER AUGMENTATION=================\n"<<std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {                                                     //for loop to run till epochs
        aug_model.train();                                                                             //training the model
        float aug_train_loss = 0.0;                                                                    //declaring a variable aug_train_loss as 0.0
        for (size_t i = 0; i < aug_X_data.size(0); i += aug_batch_size) {                               //for loop to run till size of the aug_X_data
            auto X_aug_batch = aug_X_data.narrow(0, i, std::min((size_t)(aug_batch_size), aug_X_data.size(0) - i));      //storing the batch of aug_X_train elements in X_aug_batch
            auto y_aug_batch = aug_Y_data.narrow(0, i, std::min((size_t)(aug_batch_size), aug_Y_data.size(0) - i));      //storing the batch of aug_Y_train elements in Y_aug_batch

            y_aug_batch = y_aug_batch.view({-1, output_size});                                            // reshaping the Y_aug_batch and storing in y_aug_batch

            aug_optimizer.zero_grad();                                                                    //clearing the gradients
            auto y_aug_pred = aug_model.forward(X_aug_batch);                                             //sending the X_aug_batch to the aug_model to do forward pass
            auto aug_loss = aug_loss_fn(y_aug_pred, y_aug_batch);                                         //finding the loss
            aug_loss.backward();                                                                          //doing back propagation
            aug_optimizer.step();                                                                         //updating the weights and biases

            aug_train_loss += aug_loss.item<float>();                                                     //adding the each batch loss to the aug_train_loss variable
 
        }

       aug_train_loss /= (aug_X_data.size(0) / aug_batch_size);                                           //finding the aug_train_loss for 1 epoch


        aug_model.eval();
        float aug_val_loss = 0.0;                                                                         //declaring a variable aug_val_loss as 0.0
        for (size_t j = 0; j < ValFeatures.size(0); j += aug_batch_size) {                     
            auto X_aug_batch = ValFeatures.narrow(0, j, std::min(static_cast<size_t>(aug_batch_size), ValFeatures.size(0) - j)); //storing the batch of ValFeaturs elements in X_aug_batch
            auto y_aug_batch = ValLabels.narrow(0, j, std::min(static_cast<size_t>(aug_batch_size), ValLabels.size(0) - j));     //storing the batch of ValLabels elements in y_aug_batch

            y_aug_batch = y_aug_batch.view({-1, output_size});  //reshaping the Y_aug_batch and storing in y_aug_batch

            auto y_aug_pred = aug_model.forward(X_aug_batch);   //sending the X_aug_batch to the aug_model to do forward pass
            auto aug_loss = aug_loss_fn(y_aug_pred, y_aug_batch);   //finding the loss
            aug_val_loss += aug_loss.item<float>();                 //adding the each batch loss to the aug_val_loss variable
        }
        aug_val_loss /= (ValFeatures.size(0) / aug_batch_size);     //finding the aug_val_loss for 1 epoch

        //printing the aug_train_loss and aug_val_loss for every 10 epoch
        if (epoch % 10 == 0){
            std::cout << "Epoch: " << epoch << "  Train Loss: " << aug_train_loss <<" Val loss: "<<aug_val_loss<< std::endl;
        }
    }



    aug_model.eval();                                                                                            //evaluating the model
    auto y_aug_pred = aug_model.forward(X_test);                                                                 //checking the aug_model by sending the test data
    Y_test = Y_test.view({-1, output_size});                                                                     //reshaping the Y_test
    auto aug_test_loss = loss_fn(y_aug_pred, Y_test).item<float>();                                              //finding the aug_test loss
    std::cout << "\nTest loss: " << aug_test_loss << std::endl;                                                    //printing the test loss

    auto aug_accuracy = ((y_aug_pred.round() == Y_test).to(torch::kFloat32).mean().item<float>());               //finding the accuracy of the aug_model 
    std::cout << "\nAccuracy: " << aug_accuracy << std::endl;                                                      //printing the accuracy of the model

    std::cout<<"\nPREDICTIONS:\n";
    auto aug_predictions = (y_aug_pred > 0.5).to(torch::kInt32);                                                 //printing the predictions                                         
    for (int i = 0; i < 5; i++) {                                                                                //running for loop to run till 5 elements
        std::cout << y_aug_pred[i].item<float>() << " => " << aug_predictions[i].item<int>() << " (expected " << Y_test[i].item<float>() << ")" << std::endl;  //printing the outcome
    }


    return 0;
}
