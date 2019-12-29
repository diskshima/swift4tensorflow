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

let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
let petalLengths = firstTrainFeaturesTransposed[2].scalars
let sepalLengths = firstTrainFeaturesTransposed[0].scalars

plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
