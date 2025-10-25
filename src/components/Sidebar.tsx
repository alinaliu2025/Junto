import { useNavigate } from "react-router-dom";
import { User } from "lucide-react";
import { Card } from "@/components/ui/card";
import { mockUser, mockClubs } from "@/data/mockData";

export function Sidebar() {
  const navigate = useNavigate();
  const joinedClubs = mockClubs.filter(club => mockUser.joinedClubs.includes(club.id));

  return (
    <aside className="w-72 h-full p-4 border-r border-border bg-background sticky top-0">
      <Card className="p-4 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center">
            <User className="h-6 w-6 text-muted-foreground" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm truncate">{mockUser.name}</h3>
            <p className="text-xs text-muted-foreground truncate">{mockUser.email}</p>
          </div>
        </div>
      </Card>

      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-sm">My Clubs</h4>
          <button
            onClick={() => navigate("/dashboard")}
            className="text-xs text-primary hover:underline"
          >
            View All
          </button>
        </div>

        <div className="space-y-2">
          {joinedClubs.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No clubs joined yet
            </p>
          ) : (
            joinedClubs.map(club => (
              <button
                key={club.id}
                onClick={() => navigate(`/club/${club.id}/member`)}
                className="w-full p-3 rounded-lg bg-secondary hover:bg-secondary/80 transition-base text-left"
              >
                <div className="flex items-center gap-2">
                  <span className="text-xl">{club.logo}</span>
                  <span className="text-sm font-medium truncate">{club.name}</span>
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}
