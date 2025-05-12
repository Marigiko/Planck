"use client"

import { useState } from "react"
import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DashboardNav } from "@/components/dashboard-nav"
import { DataUploader } from "@/components/data-uploader"
import { AlgorithmConfig } from "@/components/algorithm-config"
import { ResultsViewer } from "@/components/results-viewer"

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState("upload")
  const [dataUploaded, setDataUploaded] = useState(false)
  const [algorithmConfigured, setAlgorithmConfigured] = useState(false)
  const [results, setResults] = useState(null)

  return (
    <div className="flex min-h-screen flex-col bg-planck-beige">
      <header className="sticky top-0 z-50 w-full border-b border-planck-beige/20 bg-planck-beige/95 backdrop-blur supports-[backdrop-filter]:bg-planck-beige/60">
        <div className="container flex h-16 items-center">
          <div className="flex items-center gap-2">
            <div className="relative h-8 w-8">
              <Image src="/logo.png" alt="Planck Logo" fill className="object-contain" />
            </div>
            <span className="text-xl font-bold text-planck-dark">Planck</span>
          </div>
          <nav className="ml-auto flex gap-4">
            <Link href="/dashboard" className="text-sm font-medium text-planck-dark">
              Dashboard
            </Link>
            <Link href="/profile" className="text-sm font-medium text-planck-dark">
              Perfil
            </Link>
            <Link href="/" className="text-sm font-medium text-planck-dark">
              Cerrar Sesión
            </Link>
          </nav>
        </div>
      </header>
      <div className="flex flex-1">
        <aside className="hidden w-64 border-r bg-white md:block">
          <DashboardNav />
        </aside>
        <main className="flex-1 p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-planck-dark">Dashboard</h1>
            <p className="text-muted-foreground">Gestione sus datos y ejecute algoritmos cuánticos</p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="bg-white">
              <TabsTrigger value="upload" className="data-[state=active]:bg-planck-teal data-[state=active]:text-white">
                Carga de Datos
              </TabsTrigger>
              <TabsTrigger
                value="configure"
                disabled={!dataUploaded}
                className="data-[state=active]:bg-planck-teal data-[state=active]:text-white"
              >
                Configuración
              </TabsTrigger>
              <TabsTrigger
                value="results"
                disabled={!algorithmConfigured}
                className="data-[state=active]:bg-planck-teal data-[state=active]:text-white"
              >
                Resultados
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-4">
              <Card className="bg-white border-none">
                <CardHeader>
                  <CardTitle className="text-planck-dark">Carga de Datos</CardTitle>
                  <CardDescription>Cargue un archivo CSV con sus datos para procesamiento</CardDescription>
                </CardHeader>
                <CardContent>
                  <DataUploader
                    onUploadComplete={() => {
                      setDataUploaded(true)
                      setActiveTab("configure")
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="configure" className="space-y-4">
              <Card className="bg-white border-none">
                <CardHeader>
                  <CardTitle className="text-planck-dark">Configuración de Algoritmo</CardTitle>
                  <CardDescription>Seleccione y configure el algoritmo cuántico a utilizar</CardDescription>
                </CardHeader>
                <CardContent>
                  <AlgorithmConfig
                    onConfigComplete={() => {
                      setAlgorithmConfigured(true)
                      setActiveTab("results")
                      // Simulamos resultados después de la configuración
                      setResults({})
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="results" className="space-y-4">
              <Card className="bg-white border-none">
                <CardHeader>
                  <CardTitle className="text-planck-dark">Resultados</CardTitle>
                  <CardDescription>Visualice los resultados del procesamiento cuántico</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResultsViewer results={results} />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
