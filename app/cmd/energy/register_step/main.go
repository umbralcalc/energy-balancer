//go:build js && wasm

// register_step is the energy-balancer widget compiled as a WebAssembly
// module. It registers `stepSimulation` on the JS global and blocks
// forever so the Go runtime stays alive to service per-step calls from
// dexetera's runtime/worker.js.
//
// Build with the codegen-emitted app/energy/build.sh or directly:
//
//	GOOS=js GOARCH=wasm go build -o app/energy/src/main.wasm \
//	    ./app/cmd/energy/register_step
package main

import (
	"github.com/umbralcalc/dexetera/pkg/simio"
	"github.com/umbralcalc/energy-balancer/app/pkg/energydash"
)

func main() {
	simio.RegisterStep(energydash.NewConfig())
}
