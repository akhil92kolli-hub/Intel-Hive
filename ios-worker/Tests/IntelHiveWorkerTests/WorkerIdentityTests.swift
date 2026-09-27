import XCTest
@testable import IntelHiveWorker

final class WorkerIdentityTests: XCTestCase {
    func testIdentityIsStable() {
        let suite = UserDefaults(suiteName: "IntelHiveWorkerTests")!
        suite.removePersistentDomain(forName: "IntelHiveWorkerTests")
        let first = WorkerIdentity(store: suite)
        let second = WorkerIdentity(store: suite)
        XCTAssertEqual(first.workerID, second.workerID)
        XCTAssertEqual(first.publicKey, second.publicKey)
    }
}
