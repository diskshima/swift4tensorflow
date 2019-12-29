//
//  IrisBatch.swift
//  tutorial
//
//  Created by Daisuke Shimamoto on 2019/12/29.
//  Copyright © 2019 diskshima. All rights reserved.
//

import TensorFlow

/// A batch of examples from the iris dataset.
struct IrisBatch {
    /// [batchSize, featureCount] tensor of features
    let features: Tensor<Float>

    /// [batchSize] tensor of labels.
    let labels: Tensor<Int32>
}
