// Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

//  1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import { WebSocketService } from "./../../webSocket/web-socket.service";
import { Component } from "@angular/core";
import { CommonModule } from "@angular/common";

@Component({
  selector: "app-record",
  standalone: true,
  imports: [CommonModule],
  templateUrl: "./record.component.html",
  styleUrl: "./record.component.css",
})
export class RecordComponent {
  isRecording = false;
  downloadUrl: string | null = null;

  constructor(private webSocketService: WebSocketService) {}

  startRecord() {
    this.webSocketService.sendMessageToFlask(
      JSON.stringify({ Name: "DataCollection", Action: "start" }),
    );
    this.isRecording = true;
    this.downloadUrl = null;
  }

  stopRecord() {
    this.webSocketService.sendMessageToFlask(
      JSON.stringify({ Name: "DataCollection", Action: "stop" }),
    );
    this.isRecording = false;
    setTimeout(() => {
      this.downloadUrl = "/api/data-collection/latest";
    }, 2000); // Đợi backend đóng gói zip
  }

  getButtonColor() {
    if (this.isRecording === true) {
      return "#5cb85c";
    }

    return "#d9534f";
  }
}
