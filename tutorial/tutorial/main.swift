//
//  main.swift
//  tutorial
//
//  Created by Daisuke Shimamoto on 2019/12/28.
//  Copyright © 2019 diskshima. All rights reserved.
//

import Foundation
import TensorFlow
import Python
let plt = Python.import("matplotlib.pyplot")

var x = Tensor<Float>([[1, 2], [3, 4]])
print(x * 3)

func download(from sourceString: String, to destinationString: String) {
    print("Downloading \(sourceString)")
    let source = URL(string: sourceString)!

    let fm = FileManager.default
    if fm.fileExists(atPath: destinationString) {
        print("File already exists. Not downloading.")
        return
    }

    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
    print("Downloaded \(destination.lastPathComponent) to \(fm.currentDirectoryPath)")
}

download(
    from: "https://raw.githubusercontent.com/tensorflow/swift/master/docs/site/tutorials/TutorialDatasetCSVAPI.swift",
    to: "tutorial/tutorial/TutorialDatasetCSVAPI.swift")

let trainDataFilename = "iris_training.csv"
download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)

let f = Python.open(trainDataFilename)
for _ in 0..<5 {
    print(Python.next(f).strip())
}
f.close()

let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
let labelName = "species"
let columnNames = featureNames + [labelName]
print("Features: \(featureNames)")
print("Label: \(labelName)")

let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]

let batchSize = 32

let trainDataset: Dataset<IrisBatch> = Dataset(
    contentsOfCSVFile: trainDataFilename, hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)

let firstTrainExamples = trainDataset.first!
let firstTrainFeatures = firstTrainExamples.features
let firstTrainLabels = firstTrainExamples.labels
print("First batch of features: \(firstTrainFeatures)")
print("First batch of labels: \(firstTrainLabels)")

// Show scatter plot
//let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
//let petalLengths = firstTrainFeaturesTransposed[2].scalars
//let sepalLengths = firstTrainFeaturesTransposed[0].scalars

//plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
//plt.xlabel("Petal length")
//plt.ylabel("Sepal length")
//plt.show()

// Build neural network model
let hiddenSize: Int = 10
struct IrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

var model = IrisModel()

let firstTrainPredictions = model(firstTrainFeatures)
print(firstTrainPredictions[0..<5])
print(softmax(firstTrainPredictions[0..<5]))

print("Predictions: \(firstTrainPredictions.argmax(squeezingAxis: 1))")
print("     Labels: \(firstTrainLabels)")

// Loss function example
let untrainedLogits = model(firstTrainFeatures)
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: firstTrainLabels)
print("Loss test: \(untrainedLoss)")

// SGD
let optimizer = SGD(for: model, learningRate: 0.01)
let (loss, grads) = model.valueWithGradient { model -> Tensor<Float> in
    let logits = model(firstTrainFeatures)
    return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
}
print("Current loss: \(loss)")

optimizer.update(&model, along: grads)
let logitsAfterOneStep = model(firstTrainFeatures)
let lossAfterOneStep = softmaxCrossEntropy(logits: logitsAfterOneStep, labels: firstTrainLabels)
print("Next loss: \(lossAfterOneStep)")

// Training loop
let epochCount = 500
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []

func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batch in trainDataset {
        let (loss, grad) = model.valueWithGradient { (model: IrisModel) -> Tensor<Float> in
            let logits = model(batch.features)
            return softmaxCrossEntropy(logits: logits, labels: batch.labels)
        }

        optimizer.update(&model, along: grad)

        let logits = model(batch.features)
        epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
        epochLoss += loss.scalarized()
        batchCount += 1
    }

    epochAccuracy /= Float(batchCount)
    epochLoss /= Float(batchCount)
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)

    if epoch % 50 == 0 {
        print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}

// Show result plot
//plt.figure(figsize: [12, 8])
//let accuracyAxes = plt.subplot(2, 1, 1)
//accuracyAxes.set_ylabel("Accuracy")
//accuracyAxes.plot(trainAccuracyResults)
//
//let lossAxes = plt.subplot(2, 1, 2)
//lossAxes.set_ylabel("Loss")
//lossAxes.set_xlabel("Epoch")
//lossAxes.plot(trainLossResults)
//
//plt.show()

// Testing dataset
let testDataFilename = "iris_test.csv"
download(from: "http://download.tensorflow.org/data/iris_test.csv", to: testDataFilename)

let testDataset: Dataset<IrisBatch> = Dataset(
    contentsOfCSVFile: testDataFilename, hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)

for testBatch in testDataset {
    let logits = model(testBatch.features)
    let predictions = logits.argmax(squeezingAxis: 1)
    print("Test batch accuracy: \(accuracy(predictions: predictions, truths: testBatch.labels))")
}

let firstTestBatch = testDataset.first!
let firstTestBatchLogits = model(firstTestBatch.features)
let firstTestBatchPredictions = firstTestBatchLogits.argmax(squeezingAxis: 1)

print(firstTestBatchPredictions)
print(firstTestBatch.labels)

// Unlabeled dataset
let unlabledDataset: Tensor<Float> =
    [[5.1, 3.3, 1.7, 0.5],
     [5.9, 3.0, 4.2, 1.5],
     [6.9, 3.1, 5.4, 2.1]]

let unlabeledDatasetPredictions = model(unlabledDataset)

for i in 0..<unlabeledDatasetPredictions.shape[0] {
    let logits = unlabeledDatasetPredictions[i]
    let classIdx = logits.argmax().scalar!
    print("Example \(i) prediction: \(classNames[Int(classIdx)]) (\(softmax(logits)))")
}
