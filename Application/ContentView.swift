//
//  ContentView.swift
//

import SwiftUI


struct ContentView: View {
	var body: some View {
		MetalView() {
			Text("Failed to initialize renderer")
		}
#if os(iOS)
		.edgesIgnoringSafeArea(.all)
		.preferredColorScheme(.dark)
#endif
	}
}


private struct MetalView<Fallback: View> {
	let fallbackContent: () -> Fallback
}

#if os(iOS)
extension MetalView: UIViewControllerRepresentable {
	func makeUIViewController(context _: Context) -> some UIViewController {
		MetalViewController(makeFallbackChild: {
			UIHostingController(rootView: self.fallbackContent())
		})
	}

	func updateUIViewController(_: UIViewControllerType, context _: Context) {}
}
#elseif os(macOS)
extension MetalView: NSViewControllerRepresentable {
	func makeNSViewController(context _: Context) -> some NSViewController {
		MetalViewController(makeFallbackChild: {
			NSHostingController(rootView: self.fallbackContent())
		})
	}

	func updateNSViewController(_: NSViewControllerType, context _: Context) {}
}
#endif

struct ContentView_Previews: PreviewProvider {
	static var previews: some View {
		ContentView()
	}
}
