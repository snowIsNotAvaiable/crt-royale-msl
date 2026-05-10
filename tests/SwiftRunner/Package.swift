// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SwiftRunner",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "SwiftRunner", targets: ["SwiftRunner"]),
    ],
    targets: [
        .executableTarget(
            name: "SwiftRunner",
            path: "Sources/SwiftRunner"
        ),
    ]
)
