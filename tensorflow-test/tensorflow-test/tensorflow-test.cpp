// tensorflow-test.cpp : 定义控制台应用程序的入口点。
//


#include "stdafx.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "MNIST.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
using namespace chrono;
using namespace tensorflow;

int main( int argc, char* argv[] )
{

	// Initialize a tensorflow session
	cout << "start initalize session" << "\n";
	Session* session;
	Status status = NewSession( SessionOptions(), &session );
	if (!status.ok())
	{
		cout << status.ToString() << "\n";
		return 1;
	}

	// Read in the protobuf graph we exported
	// (The path seems to be relative to the cwd. Keep this in mind
	// when using `bazel run` since the cwd isn't where you call
	// `bazel run` but from inside a temp folder.)
	GraphDef graph_def;
	status = ReadBinaryProto( Env::Default(), "../graph.pb", &graph_def );
	if (!status.ok())
	{
		cout << status.ToString() << "\n";
		return 1;
	}

	// Add the graph to the session
	status = session->Create( graph_def );
	if (!status.ok())
	{
		cout << status.ToString() << "\n";
		return 1;
	}

	cout << "preparing input data..." << endl;
	// config setting
	int imageDim = 784;
	int nTests = 10000;

	// Setup inputs and outputs:
	Tensor x( DT_FLOAT, TensorShape( { nTests, imageDim } ) );

	MNIST mnist = MNIST( "../MNIST_data/" );
	auto dst = x.flat<float>().data();
	for (int i = 0; i < nTests; i++)
	{
		auto img = mnist.testData.at( i ).pixelData;
		std::copy_n( img.begin(), imageDim, dst );
		dst += imageDim;
	}

	cout << "data is ready" << endl;

	cout << x.flat<float>().data() << endl;

	Tensor keep_prob( tensorflow::DT_FLOAT, tensorflow::TensorShape() );

	keep_prob.scalar<float>()() = 1.0;

	vector<pair<string, Tensor>> inputs = { { "input", x }, { "keep_prob", keep_prob } };

	// The session will initialize the outputs
	vector<Tensor> outputs;
	// Run the session, evaluating our "softmax" operation from the graph

	status = session->Run( inputs, { "output" }, {}, &outputs );

	if (!status.ok())
	{
		cout << status.ToString() << "\n";
		return 1;
	}
	else
	{
		cout << "Success load graph !! " << "\n";
	}

	// start compute the accuracy,
	// arg_max is to record which index is the largest value after 
	// computing softmax, and if arg_max is equal to testData.label,
	// means predict correct.
	int nHits = 0;
	for (vector<Tensor>::iterator it = outputs.begin(); it != outputs.end(); ++it)
	{
		auto items = it->shaped<float, 2>( { nTests, 10 } ); // 10 represent number of class
		for (int i = 0; i < nTests; i++)
		{
			int arg_max = 0;
			float val_max = items( i, 0 );
			for (int j = 0; j < 10; j++)
			{
				if (items( i, j ) > val_max)
				{
					arg_max = j;
					val_max = items( i, j );
				}
			}
			if (arg_max == mnist.testData.at( i ).label)
			{
				nHits++;
			}
		}
	}
	float accuracy = (float) nHits / nTests;
	cout << "accuracy is : " << accuracy << "\n";

	system( "PAUSE" );

	return 0;
}
