import Foundation
#if canImport(UIKit)
import UIKit
#endif
#if canImport(Metal)
import Metal
#endif

public struct DeviceCapabilities: Sendable {
    public let osVersion: String
    public let architecture: String
    public let ramMB: UInt64
    public let gpuVendor: String
    public let gpuName: String
    public let backend: String

    public static func current() -> DeviceCapabilities {
        #if canImport(UIKit)
        let os = UIDevice.current.systemVersion
        #else
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        #endif
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            return .init(osVersion: os, architecture: "arm64", ramMB: ProcessInfo.processInfo.physicalMemory / 1_048_576,
                         gpuVendor: "Apple", gpuName: device.name, backend: "metal")
        }
        #endif
        return .init(osVersion: os, architecture: "arm64", ramMB: ProcessInfo.processInfo.physicalMemory / 1_048_576,
                     gpuVendor: "unknown", gpuName: "CPU", backend: "cpu")
    }
}
