#!/bin/bash

mkdir -p dataset/citeulike;

wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike/user_data_all_dict.p -O dataset/citeulike/user_data_all_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike/user_data_test_dict.p -O dataset/citeulike/user_data_test_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike/user_data_train_dict.p -O dataset/citeulike/user_data_train_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike/user_data_vali_dict.p -O dataset/citeulike/user_data_vali_dict.p;

mkdir -p dataset/citeulike-f;

wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike-f/cf-test-1-users.dat -O dataset/citeulike-f/cf-test-1-users.dat;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike-f/cf-train-1-users.dat -O dataset/citeulike-f/cf-train-1-users.dat;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/citeulike-f/mult.dat -O dataset/citeulike-f/mult.dat

mkdir -p dataset/tradesy;

wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/tradesy/item_features.npy -O dataset/tradesy/item_features.npy;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/tradesy/user_data_all_dict.p -O dataset/tradesy/user_data_all_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/tradesy/user_data_test_dict.p -O dataset/tradesy/user_data_test_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/tradesy/user_data_train_dict.p -O dataset/tradesy/user_data_train_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/tradesy/user_data_vali_dict.p -O dataset/tradesy/user_data_vali_dict.p

mkdir -p dataset/amazon;

wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/amazon_dataset/process/user_data_all_dict.p -O dataset/amazon/user_data_all_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/amazon_dataset/process/user_data_test_dict.p -O dataset/amazon/user_data_test_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/amazon_dataset/process/user_data_train_dict.p -O dataset/amazon/user_data_train_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/amazon_dataset/process/user_data_vali_dict.p -O dataset/amazon/user_data_vali_dict.p;
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/amazon_dataset/process/book_features_update.mem -O dataset/amazon/book_features_update.mem
wget https://s3-us-west-1.amazonaws.com/cornell-tech-sdl-immerse/amazon_dataset/process/user_features_categories.npy -O dataset/amazon/user_features_categories.npy
