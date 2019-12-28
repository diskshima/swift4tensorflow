//
//  main.swift
//  tutorial
//
//  Created by Daisuke Shimamoto on 2019/12/28.
//  Copyright © 2019 diskshima. All rights reserved.
//

import Foundation
import TensorFlow

var x = Tensor<Float>([[1, 2], [3, 4]])
print(x * 3)

func download(from sourceString: String, to destinationString: String) {
    let source = URL(string: sourceString)!
    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
}

//download(
//    from: "https://raw.githubusercontent.com/tensorflow/swift/master/docs/site/tutorials/TutorialDatasetCSVAPI.swift",
//    to: "TutorialDatasetCSVAPI.swift")
//print("File downloaded to \(FileManager.default.currentDirectoryPath)")

let trainDataFilename = "iris_training.csv"
download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)

import Python

let f = Python.open(trainDataFilename)
for _ in 0..<5 {
    print(Python.next(f).strip())
}
f.close()
