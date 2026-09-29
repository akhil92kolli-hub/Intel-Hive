// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "IntelHiveWorker",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "IntelHiveWorker", targets: ["IntelHiveWorker"])
    ],
    targets: [
        .target(name: "IntelHiveWorker"),
        .testTarget(name: "IntelHiveWorkerTests", dependencies: ["IntelHiveWorker"])
    ]
)
