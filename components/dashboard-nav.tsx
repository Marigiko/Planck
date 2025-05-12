"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Home, Upload, Settings, BarChart, Database, HelpCircle, LogOut } from "lucide-react"

export function DashboardNav() {
  return (
    <div className="flex h-full flex-col p-4">
      <div className="space-y-1">
        <h2 className="px-4 text-lg font-semibold tracking-tight text-planck-dark">Dashboard</h2>
        <nav className="flex flex-col gap-1">
          <Link href="/dashboard">
            <Button
              variant="ghost"
              className="w-full justify-start text-planck-dark hover:text-planck-teal hover:bg-planck-beige"
            >
              <Home className="mr-2 h-4 w-4" />
              Inicio
            </Button>
          </Link>
          <Link href="/dashboard?tab=upload">
            <Button
              variant="ghost"
              className="w-full justify-start text-planck-dark hover:text-planck-teal hover:bg-planck-beige"
            >
              <Upload className="mr-2 h-4 w-4" />
              Carga de Datos
            </Button>
          </Link>
          <Link href="/dashboard?tab=configure">
            <Button
              variant="ghost"
              className="w-full justify-start text-planck-dark hover:text-planck-teal hover:bg-planck-beige"
            >
              <Settings className="mr-2 h-4 w-4" />
              Configuración
            </Button>
          </Link>
          <Link href="/dashboard?tab=results">
            <Button
              variant="ghost"
              className="w-full justify-start text-planck-dark hover:text-planck-teal hover:bg-planck-beige"
            >
              <BarChart className="mr-2 h-4 w-4" />
              Resultados
            </Button>
          </Link>
        </nav>
      </div>
      <div className="mt-4 space-y-1">
        <h2 className="px-4 text-lg font-semibold tracking-tight text-planck-dark">Herramientas</h2>
        <nav className="flex flex-col gap-1">
          <Link href="/datasets">
            <Button
              variant="ghost"
              className="w-full justify-start text-planck-dark hover:text-planck-teal hover:bg-planck-beige"
            >
              <Database className="mr-2 h-4 w-4" />
              Mis Datasets
            </Button>
          </Link>
          <Link href="/help">
            <Button
              variant="ghost"
              className="w-full justify-start text-planck-dark hover:text-planck-teal hover:bg-planck-beige"
            >
              <HelpCircle className="mr-2 h-4 w-4" />
              Ayuda
            </Button>
          </Link>
        </nav>
      </div>
      <div className="mt-auto">
        <Link href="/">
          <Button variant="ghost" className="w-full justify-start text-red-500 hover:text-red-600 hover:bg-red-50">
            <LogOut className="mr-2 h-4 w-4" />
            Cerrar Sesión
          </Button>
        </Link>
      </div>
    </div>
  )
}
