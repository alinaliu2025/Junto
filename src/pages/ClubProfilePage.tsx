import { useParams, useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, MapPin, Clock, Users, Instagram, Facebook, Globe } from "lucide-react";
import { mockClubs } from "@/data/mockData";

export default function ClubProfilePage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const club = mockClubs.find((c) => c.id === id);

  if (!club) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <p className="text-muted-foreground">Club not found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1">
        {/* Hero Section */}
        <div
          className="h-64 bg-cover bg-center relative"
          style={{ backgroundImage: `url(${club.coverImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 to-black/60" />
          <Button
            variant="ghost"
            className="absolute top-4 left-4 text-white hover:bg-white/20"
            onClick={() => navigate("/discovery")}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
        </div>

        <div className="max-w-5xl mx-auto px-6 -mt-16 relative z-10 pb-12">
          <Card className="p-8 shadow-soft-lg">
            {/* Club Header */}
            <div className="flex items-start gap-6 mb-6">
              <div className="text-6xl">{club.logo}</div>
              <div className="flex-1">
                <h1 className="text-3xl font-bold mb-2">{club.name}</h1>
                <span className="inline-block px-3 py-1 text-sm rounded-full bg-secondary text-secondary-foreground">
                  {club.category}
                </span>
              </div>
              <Button variant="join" size="lg">
                Join Club
              </Button>
            </div>

            {/* Quick Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 pb-8 border-b border-border">
              <div className="flex items-center gap-3">
                <Clock className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Meeting Time</p>
                  <p className="text-sm text-muted-foreground">{club.meetingTime}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <MapPin className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Location</p>
                  <p className="text-sm text-muted-foreground">{club.location}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Users className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Members</p>
                  <p className="text-sm text-muted-foreground">{club.memberCount} active</p>
                </div>
              </div>
            </div>

            {/* Description */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4">About</h2>
              <p className="text-muted-foreground leading-relaxed">
                {club.fullDescription}
              </p>
            </div>

            {/* Social Links */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Connect With Us</h2>
              <div className="flex gap-3">
                {club.socials.instagram && (
                  <Button variant="outline" size="sm">
                    <Instagram className="h-4 w-4 mr-2" />
                    Instagram
                  </Button>
                )}
                {club.socials.facebook && (
                  <Button variant="outline" size="sm">
                    <Facebook className="h-4 w-4 mr-2" />
                    Facebook
                  </Button>
                )}
                {club.socials.website && (
                  <Button variant="outline" size="sm">
                    <Globe className="h-4 w-4 mr-2" />
                    Website
                  </Button>
                )}
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
}
