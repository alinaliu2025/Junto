import { useState } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";

interface CreateEventModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (event: { name: string; date: string; time: string; location: string; description: string }) => void;
}

export function CreateEventModal({ isOpen, onClose, onSubmit }: CreateEventModalProps) {
  const [formData, setFormData] = useState({
    name: "",
    date: "",
    time: "",
    location: "",
    description: ""
  });

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
    setFormData({ name: "", date: "", time: "", location: "", description: "" });
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-background rounded-lg shadow-soft-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-background border-b border-border p-6 flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Create New Event</h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-base"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div className="space-y-2">
            <Label htmlFor="eventName">Event Name</Label>
            <Input
              id="eventName"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., Monthly Photo Walk"
              required
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="eventDate">Date</Label>
              <Input
                id="eventDate"
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="eventTime">Time</Label>
              <Input
                id="eventTime"
                type="time"
                value={formData.time}
                onChange={(e) => setFormData({ ...formData, time: e.target.value })}
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="eventLocation">Location</Label>
            <Input
              id="eventLocation"
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
              placeholder="e.g., Memorial Union Quad"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="eventDescription">Description</Label>
            <Textarea
              id="eventDescription"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Describe the event details..."
              rows={4}
              required
            />
          </div>

          <div className="flex gap-3 justify-end pt-4">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="join">
              Create Event
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
