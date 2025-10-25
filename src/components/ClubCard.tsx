import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Users } from "lucide-react";
import { Club } from "@/data/mockData";

interface ClubCardProps {
  club: Club;
}

export function ClubCard({ club }: ClubCardProps) {
  const navigate = useNavigate();

  return (
    <Card className="overflow-hidden hover:shadow-soft-lg transition-smooth cursor-pointer group">
      <div
        onClick={() => navigate(`/club/${club.id}`)}
        className="p-6 flex flex-col h-full"
      >
        <div className="flex items-start gap-4 mb-4">
          <div className="text-4xl">{club.logo}</div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold mb-1 group-hover:text-primary transition-base">
              {club.name}
            </h3>
            <span className="inline-block px-2 py-1 text-xs rounded-full bg-secondary text-secondary-foreground">
              {club.category}
            </span>
          </div>
        </div>

        <p className="text-sm text-muted-foreground mb-4 line-clamp-2 flex-1">
          {club.shortDescription}
        </p>

        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
          <Users className="h-4 w-4" />
          <span>{club.memberCount} members</span>
        </div>

        <div className="flex gap-2">
          <Button
            variant="info"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              navigate(`/club/${club.id}`);
            }}
            className="flex-1"
          >
            More Info
          </Button>
          <Button
            variant="join"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              // Handle join action
            }}
            className="flex-1"
          >
            Join
          </Button>
        </div>
      </div>
    </Card>
  );
}
