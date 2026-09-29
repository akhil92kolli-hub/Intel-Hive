import Foundation
#if canImport(UIKit)
import UIKit
#endif
#if canImport(Metal)
import Metal
#endif
#if canImport(os)
import os
#endif
#if canImport(Network)
import Network
#endif

public struct DeviceCapabilities: Sendable {
    public let osVersion: String
    public let architecture: String
    public let ramMB: UInt64
    public let gpuVendor: String
    public let gpuName: String
    public let backend: String
    public let availableRAMMB: UInt64

    public static func current() -> DeviceCapabilities {
        #if canImport(UIKit)
        let os = UIDevice.current.systemVersion
        #else
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        #endif
        let totalRAMMB = ProcessInfo.processInfo.physicalMemory / 1_048_576
        #if os(iOS) || os(tvOS) || os(watchOS)
        let availableBytes = UInt64(os_proc_available_memory())
        let availableRAMMB = Swift.min(totalRAMMB, availableBytes / 1_048_576)
        #else
        let availableRAMMB = totalRAMMB
        #endif
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            return .init(
                osVersion: os,
                architecture: "arm64",
                ramMB: totalRAMMB,
                gpuVendor: "Apple",
                gpuName: device.name,
                backend: "metal",
                availableRAMMB: availableRAMMB
            )
        }
        #endif
        return .init(
            osVersion: os,
            architecture: "arm64",
            ramMB: totalRAMMB,
            gpuVendor: "unknown",
            gpuName: "CPU",
            backend: "cpu",
            availableRAMMB: availableRAMMB
        )
    }

    public static func powerStatus() -> PowerInfo {
        #if canImport(UIKit)
        let device = UIDevice.current
        let wasMonitoringBattery = device.isBatteryMonitoringEnabled
        device.isBatteryMonitoringEnabled = true
        defer { device.isBatteryMonitoringEnabled = wasMonitoringBattery }

        let level = device.batteryLevel
        let batteryPercent = level >= 0 ? UInt32((level * 100).rounded()) : 0
        let charging = device.batteryState == .charging || device.batteryState == .full
        return PowerInfo(batteryPercent: batteryPercent, charging: charging)
        #else
        return PowerInfo(batteryPercent: 0, charging: false)
        #endif
    }

    public static func currentNetworkType() async -> String {
        #if canImport(Network)
        await withCheckedContinuation { continuation in
            let monitor = NWPathMonitor()
            monitor.pathUpdateHandler = { path in
                let type: String
                if path.usesInterfaceType(.wifi) {
                    type = "wifi"
                } else if path.usesInterfaceType(.cellular) {
                    type = "cellular"
                } else if path.usesInterfaceType(.wiredEthernet) {
                    type = "ethernet"
                } else {
                    type = "unknown"
                }
                monitor.pathUpdateHandler = nil
                monitor.cancel()
                continuation.resume(returning: type)
            }
            monitor.start(queue: DispatchQueue(label: "com.intellihive.worker.network"))
        }
        #else
        return "unknown"
        #endif
    }
}
