import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/context/ThemeContext"

export function ThemeToggle() {
  const { isDarkMode, toggleTheme } = useTheme()

  return (
    <Button
      variant="ghost"
      size="icon"
      className="text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800 transition-colors"
      onClick={toggleTheme}
    >
      {isDarkMode ? (
        <Moon className="h-[1.1rem] w-[1.1rem]" />
      ) : (
        <Sun className="h-[1.1rem] w-[1.1rem]" />
      )}
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}