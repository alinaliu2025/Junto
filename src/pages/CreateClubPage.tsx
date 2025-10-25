import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Upload } from "lucide-react";
import { categories } from "@/data/mockData";
import { toast } from "sonner";

export default function CreateClubPage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    category: "",
    description: "",
    logo: "",
    instagram: "",
    website: ""
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success("Club created successfully!");
    // Navigate to a mock club dashboard
    navigate("/club/new/dashboard");
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1 p-6">
        <div className="max-w-4xl mx-auto">
          <Button
            variant="ghost"
            onClick={() => navigate(-1)}
            className="mb-6"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>

          <Card className="p-8">
            <div className="mb-8">
              <h1 className="text-3xl font-bold mb-2">Create New Club</h1>
              <p className="text-muted-foreground">
                Fill out the form below to create your club on Junto
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="clubName">Club Name *</Label>
                <Input
                  id="clubName"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., OSU Photography Club"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="category">Category *</Label>
                <Select
                  value={formData.category}
                  onValueChange={(value) => setFormData({ ...formData, category: value })}
                  required
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a category" />
                  </SelectTrigger>
                  <SelectContent>
                    {categories.filter(c => c !== "All").map((category) => (
                      <SelectItem key={category} value={category}>
                        {category}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description *</Label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Describe your club, its mission, and what activities you do..."
                  rows={6}
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="logo">Club Logo / Emoji *</Label>
                <Input
                  id="logo"
                  value={formData.logo}
                  onChange={(e) => setFormData({ ...formData, logo: e.target.value })}
                  placeholder="📷 or paste image URL"
                  required
                />
                <p className="text-xs text-muted-foreground">
                  Use an emoji or paste an image URL
                </p>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Social Media (Optional)</h3>
                
                <div className="space-y-2">
                  <Label htmlFor="instagram">Instagram Handle</Label>
                  <Input
                    id="instagram"
                    value={formData.instagram}
                    onChange={(e) => setFormData({ ...formData, instagram: e.target.value })}
                    placeholder="@yourclubname"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="website">Website URL</Label>
                  <Input
                    id="website"
                    type="url"
                    value={formData.website}
                    onChange={(e) => setFormData({ ...formData, website: e.target.value })}
                    placeholder="https://yourclub.com"
                  />
                </div>
              </div>

              <div className="flex gap-3 justify-end pt-6">
                <Button type="button" variant="outline" onClick={() => navigate(-1)}>
                  Cancel
                </Button>
                <Button type="submit" variant="join">
                  Create Club
                </Button>
              </div>
            </form>
          </Card>
        </div>
      </main>
    </div>
  );
}
