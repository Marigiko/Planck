"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface AlgorithmConfigProps {
  onConfigComplete: () => void
}

export function AlgorithmConfig({ onConfigComplete }: AlgorithmConfigProps) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("qsvm")
  const [isProcessing, setIsProcessing] = useState(false)

  const handleProcess = () => {
    setIsProcessing(true)
    // Simulamos el procesamiento
    setTimeout(() => {
      setIsProcessing(false)
      onConfigComplete()
    }, 2000)
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label>Seleccione un algoritmo cuántico</Label>
        <RadioGroup
          value={selectedAlgorithm}
          onValueChange={setSelectedAlgorithm}
          className="grid grid-cols-1 gap-4 md:grid-cols-3"
        >
          <div>
            <RadioGroupItem value="qsvm" id="qsvm" className="peer sr-only" />
            <Label
              htmlFor="qsvm"
              className="flex flex-col items-center justify-between rounded-md border-2 border-planck-teal/20 bg-planck-beige p-4 hover:bg-planck-teal/10 hover:text-planck-teal peer-data-[state=checked]:border-planck-teal [&:has([data-state=checked])]:border-planck-teal"
            >
              <span className="font-semibold">QSVM</span>
              <span className="text-xs text-center mt-1">Quantum Support Vector Machine</span>
            </Label>
          </div>
          <div>
            <RadioGroupItem value="qpca" id="qpca" className="peer sr-only" />
            <Label
              htmlFor="qpca"
              className="flex flex-col items-center justify-between rounded-md border-2 border-planck-teal/20 bg-planck-beige p-4 hover:bg-planck-teal/10 hover:text-planck-teal peer-data-[state=checked]:border-planck-teal [&:has([data-state=checked])]:border-planck-teal"
            >
              <span className="font-semibold">QPCA</span>
              <span className="text-xs text-center mt-1">Quantum Principal Component Analysis</span>
            </Label>
          </div>
          <div>
            <RadioGroupItem value="qaoa" id="qaoa" className="peer sr-only" />
            <Label
              htmlFor="qaoa"
              className="flex flex-col items-center justify-between rounded-md border-2 border-planck-teal/20 bg-planck-beige p-4 hover:bg-planck-teal/10 hover:text-planck-teal peer-data-[state=checked]:border-planck-teal [&:has([data-state=checked])]:border-planck-teal"
            >
              <span className="font-semibold">QAOA</span>
              <span className="text-xs text-center mt-1">Quantum Approximate Optimization Algorithm</span>
            </Label>
          </div>
        </RadioGroup>
      </div>

      <Tabs defaultValue={selectedAlgorithm} value={selectedAlgorithm} onValueChange={setSelectedAlgorithm}>
        <TabsList className="bg-planck-beige">
          <TabsTrigger value="qsvm" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            QSVM
          </TabsTrigger>
          <TabsTrigger value="qpca" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            QPCA
          </TabsTrigger>
          <TabsTrigger value="qaoa" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            QAOA
          </TabsTrigger>
        </TabsList>

        <TabsContent value="qsvm" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="pt-6 space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="kernel-param">Parámetro de Kernel</Label>
                  <span className="text-sm">0.5</span>
                </div>
                <Slider
                  id="kernel-param"
                  defaultValue={[0.5]}
                  max={1}
                  step={0.01}
                  className="[&>span]:bg-planck-teal"
                />
              </div>

              <div className="flex items-center space-x-2">
                <Switch id="normalize-qsvm" />
                <Label htmlFor="normalize-qsvm">Normalizar datos</Label>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="shots-qsvm">Número de shots</Label>
                  <span className="text-sm">1024</span>
                </div>
                <Slider
                  id="shots-qsvm"
                  defaultValue={[1024]}
                  min={100}
                  max={8192}
                  step={100}
                  className="[&>span]:bg-planck-teal"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="qpca" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="pt-6 space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="components">Componentes Principales</Label>
                  <span className="text-sm">2</span>
                </div>
                <Slider
                  id="components"
                  defaultValue={[2]}
                  min={1}
                  max={10}
                  step={1}
                  className="[&>span]:bg-planck-teal"
                />
              </div>

              <div className="flex items-center space-x-2">
                <Switch id="normalize-qpca" defaultChecked />
                <Label htmlFor="normalize-qpca">Normalizar datos</Label>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="shots-qpca">Número de shots</Label>
                  <span className="text-sm">1024</span>
                </div>
                <Slider
                  id="shots-qpca"
                  defaultValue={[1024]}
                  min={100}
                  max={8192}
                  step={100}
                  className="[&>span]:bg-planck-teal"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="qaoa" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="pt-6 space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="p-param">Parámetro p</Label>
                  <span className="text-sm">1</span>
                </div>
                <Slider id="p-param" defaultValue={[1]} min={1} max={5} step={1} className="[&>span]:bg-planck-teal" />
              </div>

              <div className="flex items-center space-x-2">
                <Switch id="use-cobyla" defaultChecked />
                <Label htmlFor="use-cobyla">Usar optimizador COBYLA</Label>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="max-iter">Iteraciones máximas</Label>
                  <span className="text-sm">100</span>
                </div>
                <Slider
                  id="max-iter"
                  defaultValue={[100]}
                  min={10}
                  max={500}
                  step={10}
                  className="[&>span]:bg-planck-teal"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Button
        onClick={handleProcess}
        disabled={isProcessing}
        className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white"
      >
        {isProcessing ? "Procesando..." : "Procesar Datos"}
      </Button>
    </div>
  )
}
