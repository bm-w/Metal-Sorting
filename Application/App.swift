//
//  App.swift
//

import SwiftUI


@main struct App: SwiftUI.App {
#if os(macOS)
	@NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
#endif

	var body: some Scene {
		WindowGroup {
			ContentView()
		}
	}
}

#if os(macOS)
class AppDelegate: NSObject, NSApplicationDelegate {
	func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
#endif


// MARK: - Utilities

func asserted<T>(_ value: T?) -> T? { assert(value != nil); return value }
