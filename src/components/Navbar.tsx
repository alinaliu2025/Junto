import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { LayoutDashboard, Search, LogOut, User } from "lucide-react";

export function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleSignOut = () => {
    navigate("/");
  };

  // Don't show navbar on sign-in page
  if (location.pathname === "/") {
    return null;
  }

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-8">
          <button
            onClick={() => navigate("/discovery")}
            className="text-2xl font-bold tracking-tight hover:text-primary transition-base"
          >
            Junto
          </button>

          <div className="hidden md:flex items-center gap-4">
            <Button
              variant={location.pathname === "/discovery" ? "default" : "ghost"}
              size="sm"
              onClick={() => navigate("/discovery")}
            >
              <Search className="h-4 w-4 mr-2" />
              Discover
            </Button>
            <Button
              variant={location.pathname === "/dashboard" ? "default" : "ghost"}
              size="sm"
              onClick={() => navigate("/dashboard")}
            >
              <LayoutDashboard className="h-4 w-4 mr-2" />
              Dashboard
            </Button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button 
            variant="ghost" 
            size="icon"
            onClick={() => navigate("/profile")}
            className="rounded-full"
          >
            <User className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={handleSignOut}>
            <LogOut className="h-4 w-4 mr-2" />
            Sign Out
          </Button>
        </div>
      </div>
    </nav>
  );
}
