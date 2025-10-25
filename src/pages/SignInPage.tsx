import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";

export default function SignInPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSignIn = (e: React.FormEvent) => {
    e.preventDefault();
    // UI only - no actual authentication
    navigate("/discovery");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md p-8 shadow-soft-lg">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">Junto</h1>
          <p className="text-muted-foreground">
            Discover and connect with OSU clubs
          </p>
        </div>

        <form onSubmit={handleSignIn} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium mb-2">
              Email
            </label>
            <Input
              id="email"
              type="email"
              placeholder="you@oregonstate.edu"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="rounded-lg"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium mb-2">
              Password
            </label>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="rounded-lg"
            />
          </div>

          <Button type="submit" className="w-full" size="lg">
            Sign In
          </Button>
        </form>

        <p className="text-xs text-center text-muted-foreground mt-6">
          By signing in, you agree to Junto's terms and conditions
        </p>
      </Card>
    </div>
  );
}
