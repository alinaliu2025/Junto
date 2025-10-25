import { useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Clock, MapPin } from "lucide-react";
import { mockUser, mockClubs } from "@/data/mockData";

export default function MemberDashboardPage() {
  const navigate = useNavigate();
  const joinedClubs = mockClubs.filter((club) =>
    mockUser.joinedClubs.includes(club.id)
  );

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">My Clubs</h1>
              <p className="text-muted-foreground">
                Access your joined clubs and stay updated
              </p>
            </div>
            <Button variant="join" onClick={() => navigate("/create-club")}>
              Create New Club
            </Button>
          </div>

          {joinedClubs.length === 0 ? (
            <Card className="p-12 text-center">
              <p className="text-muted-foreground mb-4">
                You haven't joined any clubs yet
              </p>
              <button
                onClick={() => navigate("/discovery")}
                className="text-primary hover:underline"
              >
                Discover clubs to join
              </button>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {joinedClubs.map((club) => (
                <Card
                  key={club.id}
                  className="overflow-hidden hover:shadow-soft-lg transition-smooth cursor-pointer group"
                  onClick={() => navigate(`/club/${club.id}/member`)}
                >
                  <div
                    className="h-32 bg-cover bg-center"
                    style={{ backgroundImage: `url(${club.coverImage})` }}
                  >
                    <div className="h-full bg-gradient-to-b from-black/20 to-black/60 flex items-end p-4">
                      <div className="flex items-center gap-3">
                        <span className="text-3xl">{club.logo}</span>
                        <h3 className="text-white font-semibold text-lg">
                          {club.name}
                        </h3>
                      </div>
                    </div>
                  </div>

                  <div className="p-4 space-y-3">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="h-4 w-4" />
                      <span>{club.meetingTime}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <MapPin className="h-4 w-4" />
                      <span className="truncate">{club.location}</span>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
