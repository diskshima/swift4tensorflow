# Swift for Tensorflow Practice

Based on [Model training walkthrough  |  Swift for TensorFlow](https://www.tensorflow.org/swift/tutorials/model_training_walkthrough) but adjusted to work inside Xcode.

## Running Swift for Tensorflow via xcrun

1. Install Swift for Tensorflow from [here](https://github.com/tensorflow/swift/blob/master/Installation.md).
1. Find the Bundle Identifier of the toolchain installed above.
    ```bash
    $ cat /Library/Developer/Toolchains/swift-tensorflow-RELEASE-0.6.xctoolchain/Info.plist
        :
            <key>CFBundleIdentifier</key>
            <string>com.google.swift.20191121</string>    <-- This is the one you need.
        :
    ```
1. Run xcrun specifying the toolchain.
    ```bash
    $ xcrun --toolchain "com.google.swift.20191121" swift --version
    Swift version 5.1.1-dev (Swift 7b97b0ced0)
    Target: x86_64-apple-darwin19.2.0
    ```
1. Try out the test script.
    ```bash
    $ xcrun --toolchain "com.google.swift.20191121" swift test.swift
    [[ 3.0,  6.0],
     [ 9.0, 12.0]]
    ```
