package com.example.segapp

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.graphics.drawable.toBitmap
import com.example.segapp.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var imgOriginal: android.widget.ImageView
    private lateinit var imgOverlay: android.widget.ImageView
    private lateinit var tvInfo: android.widget.TextView
    private var pickedUri: Uri? = null
    private val unet by lazy { TFLiteUNet(this, size = 256) }

    private val pickImage = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            try {
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: SecurityException) { /* emulador sem persist */ }
            pickedUri = uri
            val bmp = loadBitmapFromUri(uri)  // já vem em ARGB_8888 "software"
            imgOriginal.setImageBitmap(bmp)
            tvInfo.text = "Imagem carregada."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val vb = ActivityMainBinding.inflate(layoutInflater)
        setContentView(vb.root)

        imgOriginal = vb.imgOriginal
        imgOverlay = vb.imgOverlay
        tvInfo = vb.tvInfo

        vb.btnPick.setOnClickListener {
            pickImage.launch(arrayOf("image/*"))
        }

        vb.btnSegment.setOnClickListener {
            // garante ARGB_8888 também se vier de um Drawable
            val bmp: Bitmap? = imgOriginal.drawable?.toBitmap(config = Bitmap.Config.ARGB_8888)
            if (bmp == null) {
                tvInfo.text = "Selecione uma imagem primeiro."
                return@setOnClickListener
            }
            tvInfo.text = "Executando inferência…"
            Thread {
                try {
                    val out = unet.segment(bmp)
                    runOnUiThread {
                        imgOverlay.setImageBitmap(out)
                        tvInfo.text = "Ok! Máscara sobreposta."
                    }
                } catch (e: Exception) {
                    runOnUiThread { tvInfo.text = "Erro: ${e.message}" }
                }
            }.start()
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= 28) {
            val src = ImageDecoder.createSource(contentResolver, uri)
            val bmp = ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                decoder.isMutableRequired = true
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
            bmp.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media
                .getBitmap(contentResolver, uri)
                .copy(Bitmap.Config.ARGB_8888, true)
        }
    }
}
