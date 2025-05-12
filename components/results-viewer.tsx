"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Share2 } from "lucide-react"
import { BenchmarkChart } from "@/components/benchmark-chart"

interface ResultsViewerProps {
  results: any
}

export function ResultsViewer({ results }: ResultsViewerProps) {
  const [activeView, setActiveView] = useState("chart")

  // Datos simulados para la demostración
  const accuracy = 0.87
  const f1Score = 0.85
  const precision = 0.89
  const recall = 0.82

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="bg-planck-beige border-none">
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-planck-teal">{(accuracy * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Precisión</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-planck-beige border-none">
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-planck-teal">{(f1Score * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">F1 Score</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-planck-beige border-none">
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-planck-teal">{(precision * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Precisión</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-planck-beige border-none">
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-planck-teal">{(recall * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Recall</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeView} onValueChange={setActiveView}>
        <TabsList className="bg-planck-beige">
          <TabsTrigger value="chart" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            Gráfico
          </TabsTrigger>
          <TabsTrigger value="matrix" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            Matriz de Confusión
          </TabsTrigger>
          <TabsTrigger value="data" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            Datos
          </TabsTrigger>
          <TabsTrigger value="benchmark" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
            Benchmark
          </TabsTrigger>
        </TabsList>

        <TabsContent value="chart" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="p-6">
              <div className="flex justify-center items-center h-64 bg-white rounded-md">
                <p className="text-muted-foreground">Visualización del gráfico de resultados</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="matrix" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="p-6">
              <div className="flex justify-center items-center h-64">
                <div className="grid grid-cols-2 gap-1">
                  <div className="bg-green-100 p-6 flex items-center justify-center">
                    <span className="font-bold">85</span>
                  </div>
                  <div className="bg-red-100 p-6 flex items-center justify-center">
                    <span className="font-bold">12</span>
                  </div>
                  <div className="bg-red-100 p-6 flex items-center justify-center">
                    <span className="font-bold">8</span>
                  </div>
                  <div className="bg-green-100 p-6 flex items-center justify-center">
                    <span className="font-bold">95</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="p-6">
              <div className="h-64 overflow-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="p-2 text-left">Índice</th>
                      <th className="p-2 text-left">Valor Real</th>
                      <th className="p-2 text-left">Predicción</th>
                      <th className="p-2 text-left">Probabilidad</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: 10 }).map((_, i) => (
                      <tr key={i} className="border-b">
                        <td className="p-2">{i + 1}</td>
                        <td className="p-2">{Math.random() > 0.5 ? 1 : 0}</td>
                        <td className="p-2">{Math.random() > 0.5 ? 1 : 0}</td>
                        <td className="p-2">{(Math.random() * 0.5 + 0.5).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="benchmark" className="space-y-4">
          <Card className="bg-planck-beige border-none">
            <CardContent className="p-6">
              <div className="h-64">
                <BenchmarkChart />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="flex flex-col space-y-2 sm:flex-row sm:space-y-0 sm:space-x-2">
        <Button className="flex-1 bg-planck-teal hover:bg-planck-teal/90 text-white">
          <Download className="mr-2 h-4 w-4" />
          Descargar Resultados
        </Button>
        <Button variant="outline" className="flex-1 border-planck-teal text-planck-teal hover:bg-planck-teal/10">
          <Share2 className="mr-2 h-4 w-4" />
          Compartir Resultados
        </Button>
      </div>
    </div>
  )
}
